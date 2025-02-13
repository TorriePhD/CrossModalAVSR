# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import numpy
import torch
import torch.nn.functional as F
import torch.nn as nn
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import (
    make_non_pad_mask,
    th_accuracy,
)
from espnet.nets.pytorch_backend.whisper.modeling_whisper import WhisperForConditionalGeneration, shift_tokens_right
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.nets_utils import MLPHead
import torch
from espnet.nets.pytorch_backend.backbones.conv1d_extractor import Conv1dResNet
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict

from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from fairseq.checkpoint_utils import load_model_ensemble
class E2E(torch.nn.Module):
    def __init__(self, odim, args, ignore_id=-1):
        torch.nn.Module.__init__(self)
        # if args is a dictionary create two encoders, one for audio and one for video
        self.crossmodal = False
        self.blank = 0
        
        self.odim = odim
        self.ignore_id = ignore_id
        if isinstance(args, dict):
            self.crossmodal = True
            #share the encoder layers between audio and video
            torch_dtype = torch.float16 
            whisperModel = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en",attn_implementation="flash_attention_2", torch_dtype=torch_dtype,use_safetensors=True)
            whisperModel.train()
            whisperModel.to("cuda")
            self.sos = whisperModel.config.decoder_start_token_id
            self.eos = whisperModel.config.pad_token_id
            whispConv1 = whisperModel.model.encoder.conv1
            whispConv2 = whisperModel.model.encoder.conv2
            self.audioFrontEnd = torch.nn.Sequential(
                whispConv1,
                torch.nn.GELU(),
                whispConv2,
                torch.nn.GELU()
            )
            whisperModel.model.encoder.conv1 = None
            whisperModel.model.encoder.conv2 = None
            if args["audio_backbone"].freezeAudioFrontEnd:
                for param in self.audioFrontEnd.parameters():
                    param.requires_grad = False
                self.audioFrontEnd.requires_grad = False
            else:
                self.audioFrontEnd.requires_grad = True
                for param in self.audioFrontEnd.parameters():
                    param.requires_grad = True
            self.videoFrontEnd = Conv3dResNet(relu_type = args["visual_backbone"].relu_type)
            #make videoFrontEnd  dtype float16
            #make a projection layer to prject the video features to the same shape as the audioFeatures
            self.videoProjection = nn.Linear(512, whispConv2.out_channels, dtype=torch_dtype)
            self.adim = args["audio_backbone"].adim
            self.encoder = whisperModel.model.encoder
            if args["visual_backbone"].freezeEncoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.requires_grad = False
            else:
                self.encoder.requires_grad = True
                for param in self.encoder.parameters():
                    param.requires_grad = True
                
            self.decoder = whisperModel.model.decoder
            if args["visual_backbone"].freezeDecoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
                self.decoder.requires_grad = False
            else:
                self.decoder.requires_grad = True
                for param in self.decoder.parameters():
                    param.requires_grad = True
            self.proj_out = whisperModel.proj_out
            if args["visual_backbone"].freezeDecoder:
                for param in self.proj_out.parameters():
                    param.requires_grad = False
                self.proj_out.requires_grad = False
            else:
                self.proj_out.requires_grad = True
                for param in self.proj_out.parameters():
                    param.requires_grad = True
            self.fusion = MLPHead(
                idim=384+384,
                hdim = args["fusion"].fusion_hdim,
                odim=384,
                norm=args["fusion"].fusion_norm,
            )
            self.criterion = LabelSmoothingLoss(
                self.odim,
                self.ignore_id,
                args["visual_backbone"].lsm_weight,
                args["visual_backbone"].transformer_length_normalized_loss,
            )
            self.adim = args["visual_backbone"].adim
            self.mtlalpha = args["visual_backbone"].mtlalpha
            self.modalities = ["video", "audio", "audiovisual"]
            if args["visual_backbone"].mtlalpha > 0.0:
                self.ctc = CTC(
                    odim, args["visual_backbone"].adim, args["visual_backbone"].dropout_rate, ctc_type=args["visual_backbone"].ctc_type, reduce=True
                )
            else:
                self.ctc = None   
    def forward_audio_reconstruction(self, audios: torch.Tensor) -> torch.Tensor:
        # add extra 640 padding for audios to prevent mismatch error. Extra margin will be truncated later
        extra_padding = torch.zeros(audios.size(0), 8000).to(audios.device) # 0.5 sec
        audios = torch.cat([audios, extra_padding], axis=-1)
        if self.codec == None or "vq" in self.codec.lower():
            audio_tokens = self.wav2vec.feature_extractor(audios)
            audio_tokens = self.wav2vec.vector_quantizer.forward_idx(audio_tokens)[1]
            return audio_tokens
        elif "wav2vec2" in self.codec.lower():
            # extract features from raw waveform.
            feats = self.wav2vec.wav2vec2.feature_extractor(audios).transpose(1, 2)
            _, feats = self.wav2vec.wav2vec2.feature_projection(feats)
            indices = self.wav2vec.quantizer(feats)[0].unflatten(-1, (2, -1))[..., 0].long()
            return indices

    def forward(self, x, lengths, label):
        if self.crossmodal:
            return self.forward_crossmodal(x, lengths, label)
        # if self.transformer_input_layer == "conv1d":
        #     lengths = torch.div(lengths, 640, rounding_mode="trunc")
        # padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        # x, _ = self.encoder(x, padding_mask)

        # # ctc loss
        # loss_ctc, ys_hat = self.ctc(x, lengths, label)

        # if self.proj_decoder:
        #     x = self.proj_decoder(x)

        # # decoder loss
        # ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        # ys_mask = target_mask(ys_in_pad, self.ignore_id)
        # # decoder_outputs = self.decoder(
        # #     input_ids=decoder_input_ids,
        # #     attention_mask=decoder_attention_mask,
        # #     encoder_hidden_states=encoder_outputs[0],
        # #     head_mask=decoder_head_mask,
        # #     cross_attn_head_mask=cross_attn_head_mask,
        # #     past_key_values=past_key_values,
        # #     inputs_embeds=decoder_inputs_embeds,
        # #     position_ids=decoder_position_ids,
        # #     use_cache=use_cache,
        # #     output_attentions=output_attentions,
        # #     output_hidden_states=output_hidden_states,
        # #     return_dict=return_dict,
        # # )
        # # pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, padding_mask)
        # pred_pad = self.decoder(
        #     input_ids=ys_in_pad,
        #     attention_mask=ys_mask,
        #     encoder_hidden_states=x,
        #     head_mask=None,
        #     cross_attn_head_mask=None,
        #     past_key_values=None,
        #     inputs_embeds=None,
        #     position_ids=None,
        #     use_cache=None,
        #     output_attentions=None,
        #     output_hidden_states=None,
        #     return_dict=None,
        # )
        # loss_att = self.criterion(pred_pad, ys_out_pad)
        # loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att

        # acc = th_accuracy(
        #     pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        # )

        # return loss, loss_ctc, loss_att, acc
    def getAudioFeatures(self, audio, vidSize,padding_mask=None,):
        if len(audio.size()) == 4:
            audio = audio.squeeze(1)
        xAudio = self.audioFrontEnd(audio)
        mask = padding_mask
        xAudio = self.encoder(xAudio)
        xAudio = xAudio[0]
        size = vidSize
        #padd xAud to match size of xVid
        # if size is not None:
        #     xAudio = torch.nn.functional.pad(xAudio, (0, 0, 0, size - xAudio.size(1)), "constant")
        return xAudio
    def getVideoFeatures(self, video, padding_mask=None):
        xVideo = self.videoFrontEnd(video)
        xVideo = self.videoProjection(xVideo)
        mask =  padding_mask
        if xVideo.size(1) != 1500:
            xVideo = torch.nn.functional.pad(xVideo, (0, 0, 0, 1500 - xVideo.size(1)), "constant")
        xVideo = xVideo.permute(0, 2, 1)        
        xVideo = self.encoder(xVideo)
        xVideo = xVideo[0]
        return xVideo
    def getCombinedFeatures(self, xVideo, xAudio):
        x_combined = torch.cat((xVideo, xAudio), dim=2)
        x_combined = self.fusion(x_combined)
        return x_combined
    def getSingleModalFeatures(self, video, audio, modality, padding_mask,vidSize=None,):
        if modality == "video":
            myPaddingMask = None
            if padding_mask is not None:
                myPaddingMask = padding_mask["video"]
            xVid = self.getVideoFeatures(video, myPaddingMask)
            return xVid
        elif modality == "audio":
            myPaddingMask = None
            if padding_mask is not None:
                myPaddingMask = padding_mask["audio"]

            xAud = self.getAudioFeatures(audio, vidSize, myPaddingMask)
            return xAud
        else:
            if padding_mask is None:
                padding_mask = {}
                padding_mask["video"] = None
                padding_mask["audio"] = None
            xVid = self.getVideoFeatures(video, padding_mask["video"])
            xAud = self.getAudioFeatures(audio, video.size(1), padding_mask["audio"])
            x_combined = self.getCombinedFeatures(xVid, xAud)
            return x_combined
    def getAllModalFeatures(self,x,lengths=None,label=None):
        #get audioOnlyMask as indicated where x['video'] is all zeros
        audioOnlyMask = x['video'].sum(dim=(1,2,3,4)) == 0
        videoOnlyMask = torch.all(torch.isclose(x['audio'], torch.zeros_like(x['audio']), atol=1e-11), dim=1)
        videoOtherMask = torch.all(torch.isclose(x['audio'], torch.zeros_like(x['audio']), atol=1e-11), dim=1)
        videoOtherMask = ~videoOtherMask
        otherMask = x['video'].sum(dim=(1,2,3,4)) != 0
        audioOnlyCount = audioOnlyMask.sum()
        videoOnlyCount = videoOnlyMask.sum()
        videoOtherCount = videoOtherMask.sum()
        otherCount = otherMask.sum()
        audioOnlyBatch = False
        videoOnlyBatch = False
        if videoOtherCount == 0:
            videoOnlyBatch = True
        if otherCount == 0:
            audioOnlyBatch = True
            otherMask[0] = True
            otherCount += 1
            audioOnlyCount -= 1
        if lengths is not None:
            padding_mask = {}
            for key in lengths.keys():
                myLengths = lengths[key]
                if key == "audio":
                    myLengths = torch.div(lengths[key], 640, rounding_mode="trunc")
                padding_mask[key] = make_non_pad_mask(myLengths).to(x[key].device).unsqueeze(-2)
            padding_mask["video"] = padding_mask["video"][otherMask]
            lengths["video"] = lengths["video"][otherMask]
        else:
            padding_mask = None
            
        if otherCount == 0:
            vidSize = torch.div(x["audio"].size(1), 640, rounding_mode="trunc")
        else:
            vidSize = x["video"].size(1)
        enc_feat = torch.zeros(audioOnlyCount+otherCount*3, 1500, 384, device=x["video"].device,dtype=torch.float16)
        modalities = torch.cat((torch.zeros(otherCount, dtype=torch.long, device=x["video"].device),
                                torch.ones(otherCount+audioOnlyCount, dtype=torch.long, device=x["video"].device),
                                torch.ones(otherCount, dtype=torch.long, device=x["video"].device)+1))
        for modality in ["audio", "video"]:
            if modality == "audio":
                if videoOnlyBatch:
                    continue
                indexes = modalities == 1
                video = None
                audio = x["audio"].clone()
            elif modality == "video":
                indexes = modalities == 0
                video = x["video"][otherMask].clone()
                audio = None
            if sum(indexes) == 0:
                continue
            enc_feat[indexes] = self.getSingleModalFeatures(video, audio, modality, padding_mask, vidSize, )
        videoFeat = enc_feat[modalities==0].clone()
        if not videoOnlyBatch:
            audioFeat = enc_feat[modalities==1].clone()[otherMask]
        indexes = modalities == 2
        if sum(indexes) > 0 and not videoOnlyBatch:
            enc_feat[indexes] = self.getCombinedFeatures(videoFeat, audioFeat)
        # ctc loss
        #repeat label 3 times
        if vidSize is None:
            vidSize = enc_feat.size(1)
        if label is not None:
            otherLabels = label[otherMask]
            if audioOnlyBatch:
                otherLabels[0] = 0
            label = torch.cat((otherLabels,label,otherLabels), dim=0) #resume here
            
        if lengths is not None:
            total = lengths["video"].size()[0]+lengths["audio"].size()[0]+lengths["video"].size()[0]
            #repeat vidSize total times
            lengths["video"] = torch.tensor([vidSize]).repeat(total)
            # lengths["audio"] = torch.cat((lengths["audio"], lengths["video"],lengths["video"]), dim=0)
            
            padding_mask["video"] = make_non_pad_mask(lengths["video"]).to(x["video"].device).unsqueeze(-2)
            # padding_mask["video"] = torch.cat(( padding_mask["video"],padding_mask["video"],padding_mask["video"]), dim=0)
            # padding_mask["audio"] = torch.cat((padding_mask["audio"], padding_mask["video"],padding_mask["video"]), dim=0)
        return enc_feat, lengths, padding_mask, label, modalities,otherMask
    def forward_crossmodal(self, x, lengths, label):
        enc_feat, lengths, padding_mask, label, modalities,audio_visualMask = self.getAllModalFeatures(x,lengths,label)
        if self.mtlalpha > 0:
            loss_ctcMod, ys_hat = self.ctc(enc_feat, lengths["video"], label)
            loss_ctc = loss_ctcMod
        else:
            loss_ctc = 0
        
        # decoder loss
        print("stuf",self.sos, self.eos, self.ignore_id)
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.eos)
        label =label.squeeze(1)
        #add a single -1 token to the end of label on the last dimension
        fill = torch.ones((label.size(0),1), dtype=torch.long, device=label.device)*-1
        label = torch.cat((label, fill), dim=1)

        ys_in_pad = shift_tokens_right(label, self.eos,self.sos)
        ys_mask = target_mask(ys_in_pad, self.eos)
        #change ys_mask to be float16
        ys_mask = ys_mask.to(torch.float16)
        ys_out_pad.masked_fill_(ys_out_pad == -1, self.eos)
        print(f"ys_in_pad: {ys_in_pad}, ys_out_pad: {ys_out_pad}, ys_in_pad shape: {ys_in_pad.shape}, ys_out_pad shape: {ys_out_pad.shape}, ys_in_pad dtype: {ys_in_pad.dtype}, ys_out_pad dtype: {ys_out_pad.dtype}")
        print(f"ys_mask: {ys_mask}, ys_mask shape: {ys_mask.shape}, ys_mask dtype: {ys_mask.dtype}")
        if self.mtlalpha < 1:
            decoderOutputs = self.decoder(
            input_ids=ys_in_pad,
            attention_mask=ys_mask,
            encoder_hidden_states=enc_feat,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            position_ids=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
            decoderOut = decoderOutputs.last_hidden_state 
            pred_pad = self.proj_out(decoderOut)
        else:
            pred_pad = None
        print(f"pred_pad.shape: {pred_pad.shape}, ys_out_pad.shape: {ys_out_pad.shape}")
        print(f"pred_pad: {pred_pad}, ys_out_pad: {ys_out_pad}")
        loss_att = self.criterion(pred_pad, ys_out_pad)
        if self.mtlalpha > 0:
            loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att #If it is a audio only batch, make the visual and audiovisual loss lower!!!
        else:
            loss = loss_att
        accAll = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )
        acc = {}
        if sum(modalities==0) > 0:
            acc["video"] = th_accuracy(
                pred_pad[modalities==0].view(-1, self.odim), ys_out_pad[modalities==0], ignore_label=self.ignore_id
            )
        if sum(modalities==1) > 0:
            acc["audio"] = th_accuracy(
                pred_pad[modalities==1].view(-1, self.odim), ys_out_pad[modalities==1], ignore_label=self.ignore_id
            )
        if sum(modalities==2) > 0:
            acc["audiovisual"] = th_accuracy(
                pred_pad[modalities==2].view(-1, self.odim), ys_out_pad[modalities==2], ignore_label=self.ignore_id
            )
        print(loss, loss_ctc, loss_att, accAll, acc)
        return loss, loss_ctc, loss_att, accAll, acc
