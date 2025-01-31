# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import numpy
import torch
import torch.nn.functional as F
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import (
    make_non_pad_mask,
    th_accuracy,
)
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
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        if isinstance(args, dict):
            self.crossmodal = True
            # self.audioEncoder = self.createEncoder(args["audio_backbone"])
            # self.videoEncoder = self.createEncoder(args["visual_backbone"])
            #share the encoder layers between audio and video
            # self.audioEncoder.encoders = self.videoEncoder.encoders
            self.audioFrontEnd = Conv1dResNet(relu_type = args["audio_backbone"].relu_type ,a_upsample_ratio = args["audio_backbone"].a_upsample_ratio)
            self.videoFrontEnd = Conv3dResNet(relu_type = args["visual_backbone"].relu_type)
            pos_enc_class = PositionalEncoding
            if args["visual_backbone"].transformer_encoder_attn_layer_type == "rel_mha":
                pos_enc_class = RelPositionalEncoding
            self.adim = args["audio_backbone"].adim
            self.audioEmbed = torch.nn.Sequential(torch.nn.Linear(512, self.adim), pos_enc_class(self.adim, args["audio_backbone"].dropout_rate))
            self.videoEmbed = torch.nn.Sequential(torch.nn.Linear(512, self.adim), pos_enc_class(self.adim, args["visual_backbone"].dropout_rate))
            self.encoders = self.createEncoders(args["visual_backbone"])
            self.audioAfterNorm = LayerNorm(self.adim)
            self.videoAfterNorm = LayerNorm(self.adim)
            self.transformer_input_layer = {}
            self.transformer_input_layer["audio"] = args["audio_backbone"].transformer_input_layer
            self.transformer_input_layer["video"] = args["visual_backbone"].transformer_input_layer
            self.a_upsample_ratio = {}
            self.a_upsample_ratio["audio"] = args["audio_backbone"].a_upsample_ratio
            self.a_upsample_ratio["video"] = args["visual_backbone"].a_upsample_ratio
            # self.proj_decoder = {}
            # self.proj_decoder["audio"] = None
            # self.proj_decoder["video"] = None
            # if args["audio_backbone"].adim != args["audio_backbone"].ddim:
            #     self.proj_decoder["audio"] = torch.nn.Linear(args["audio_backbone"].adim, args["audio_backbone"].ddim)
            # if args["visual_backbone"].adim != args["visual_backbone"].ddim:
            #     self.proj_decoder["video"] = torch.nn.Linear(args["visual_backbone"].adim, args["visual_backbone"].ddim)
            if args["audio_backbone"].mtlalpha < 1 or args["visual_backbone"].mtalpha < 1:
                self.decoder = Decoder(
                    odim=odim,
                    attention_dim=args["visual_backbone"].ddim,
                    attention_heads=args["visual_backbone"].dheads,
                    linear_units=args["visual_backbone"].dunits,
                    num_blocks=args["visual_backbone"].dlayers,
                    dropout_rate=args["visual_backbone"].dropout_rate,
                    positional_dropout_rate=args["visual_backbone"].dropout_rate,
                    self_attention_dropout_rate=args["visual_backbone"].transformer_attn_dropout_rate,
                    src_attention_dropout_rate=args["visual_backbone"].transformer_attn_dropout_rate,
                )
            self.fusion = MLPHead(
                idim=args["audio_backbone"].adim + args["visual_backbone"].adim,
                hdim = args["fusion"].fusion_hdim,
                odim=args["audio_backbone"].adim,
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
            self.audioReconstructionModalities = ["video"]
            if args["audio_backbone"].audio_reconstruction:
                self.audioReconstructionModalities += ["audio"]
            if args["visual_backbone"].audio_visual_reconstruction:
                self.audioReconstructionModalities += ["audiovisual"]
            if args["audio_backbone"].codec is None:
                self.codec = None
            elif "vq" in args["audio_backbone"].codec.lower():
                wav2vec, metadata = load_model_ensemble(["/workspace/vq-wav2vec_kmeans.pt"])
                self.wav2vec = wav2vec[0].requires_grad_(False).eval()
                self.audio_alignment = 4
                self.audio_vocab_size = metadata.model.vq_vars # 320
                self.video_classifier = torch.nn.Linear(768, self.audio_alignment * metadata.model.vq_groups * self.audio_vocab_size) # 768 -> 4 * 2 * 320
                # self.audio_classifier = torch.nn.Linear(768, self.audio_alignment * metadata.model.vq_groups * self.audio_vocab_size) # 768 -> 4 * 2 * 320
                self.audio_weight = args["audio_backbone"].audio_weight
                self.codec = "vq"
            elif "wav2vec2" in args["audio_backbone"].codec.lower():
                # facebook/wav2vec2-large-xlsr-53 is multilingual neural audio quantizer. 
                # We used facebook/wav2vec2-large-960h for English and kehanlu/mandarin-wav2vec2 for Mandarin.
                wav2vec = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
                del wav2vec.wav2vec2.encoder # remove transformer encoder blocks
                wav2vec = wav2vec.requires_grad_(False).eval()
                codevectors = torch.arange(wav2vec.quantizer.codevectors.size(1))
                codevectors = codevectors.view(1, -1, 1).expand_as(wav2vec.quantizer.codevectors)
                wav2vec.quantizer.codevectors.data = codevectors.float()
                self.wav2vec = wav2vec
                self.audio_alignment = 2
                self.audio_vocab_size = 640
                self.video_classifier = torch.nn.Linear(768, self.audio_alignment * 2 * self.audio_vocab_size) # 768 -> 4 * 2 * 320
                # self.audio_classifier = nn.Linear(768, self.audio_alignment * 2 * self.audio_vocab_size)
                self.audio_weight = args.audio_weight
                self.codec = "wav2vec2"
            else:
                self.codec = None
            if self.codec is not None:
                print(f"using {self.codec} neural audio codec")
            else:
                print("Inference purpose only, not using codec")
                print("To train with our method, you should set codec as 'wav2vec2' or 'vq'")             
        else:
            self.encoder = self.createEncoder(args)
            self.transformer_input_layer = args.transformer_input_layer
            self.a_upsample_ratio = args.a_upsample_ratio

            self.proj_decoder = None
            if args.adim != args.ddim:
                self.proj_decoder = torch.nn.Linear(args.adim, args.ddim)

            if args.mtlalpha < 1:
                self.decoder = Decoder(
                    odim=odim,
                    attention_dim=args.ddim,
                    attention_heads=args.dheads,
                    linear_units=args.dunits,
                    num_blocks=args.dlayers,
                    dropout_rate=args.dropout_rate,
                    positional_dropout_rate=args.dropout_rate,
                    self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                    src_attention_dropout_rate=args.transformer_attn_dropout_rate,
                )
            else:
                self.decoder = None
            # self.lsm_weight = a
            self.criterion = LabelSmoothingLoss(
                self.odim,
                self.ignore_id,
                args.lsm_weight,
                args.transformer_length_normalized_loss,
            )

            self.adim = args.adim
            self.mtlalpha = args.mtlalpha
            if args.mtlalpha > 0.0:
                self.ctc = CTC(
                    odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
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

    def createEncoders(self, args):
        encoder_attn_layer_type = args.transformer_encoder_attn_layer_type
        attention_heads = args.aheads
        attention_dim = args.adim
        attention_dropout_rate = args.transformer_attn_dropout_rate
        zero_triu = getattr(args, "zero_triu", False)
        if encoder_attn_layer_type == "mha":
                encoder_attn_layer = MultiHeadedAttention
                encoder_attn_layer_args = (
                    attention_heads,
                    attention_dim,
                    attention_dropout_rate,
                )
        elif encoder_attn_layer_type == "rel_mha":
            encoder_attn_layer = RelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + encoder_attn_layer)
        cnn_module_kernel = args.cnn_module_kernel
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)
        num_blocks = args.elayers
        linear_units = args.eunits
        dropout_rate = args.dropout_rate
        positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        use_cnn_module = args.use_cnn_module
        normalize_before = True
        concat_after = False
        macaron_style = args.macaron_style
        positionwise_layer = PositionwiseFeedForward

        return repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                macaron_style,
            ),
        )
        
    def createEncoder(self, args):
        return Encoder(
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.eunits,
                num_blocks=args.elayers,
                input_layer=args.transformer_input_layer,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
                macaron_style=args.macaron_style,
                use_cnn_module=args.use_cnn_module,
                cnn_module_kernel=args.cnn_module_kernel,
                zero_triu=getattr(args, "zero_triu", False),
                a_upsample_ratio=args.a_upsample_ratio,
                relu_type=getattr(args, "relu_type", "swish"),
            )

    def forward(self, x, lengths, label):
        if self.crossmodal:
            return self.forward_crossmodal(x, lengths, label)
        if self.transformer_input_layer == "conv1d":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        x, _ = self.encoder(x, padding_mask)

        # ctc loss
        loss_ctc, ys_hat = self.ctc(x, lengths, label)

        if self.proj_decoder:
            x = self.proj_decoder(x)

        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)
        loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att

        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        return loss, loss_ctc, loss_att, acc
    def getAudioFeatures(self, audio, vidSize,padding_mask=None,):
        if len(audio.size()) == 2:
            audio = audio.unsqueeze(0)
        xAudio = self.audioFrontEnd(audio)
        xAudio = self.audioEmbed(xAudio)
        mask = padding_mask
        xAudio = self.encoders(xAudio, mask)
        if isinstance(xAudio, tuple):
            xAudio = xAudio[0]
        if isinstance(xAudio, tuple):
            xAudio = xAudio[0]
        xAudio = self.audioAfterNorm(xAudio)
        size = vidSize
        #padd xAud to match size of xVid
        if size is not None:
            xAudio = torch.nn.functional.pad(xAudio, (0, 0, 0, size - xAudio.size(1)), "constant")
        return xAudio
    def getVideoFeatures(self, video, padding_mask=None):
        xVideo = self.videoFrontEnd(video)
        xVideo = self.videoEmbed(xVideo)
        mask =  padding_mask
        xVideo = self.encoders(xVideo, mask)
        if isinstance(xVideo, tuple):
            xVideo = xVideo[0]
        if isinstance(xVideo, tuple):
            xVideo = xVideo[0]
        xVideo = self.videoAfterNorm(xVideo)
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
        enc_feat = torch.zeros(audioOnlyCount+otherCount*3, vidSize, self.adim, device=x["video"].device)
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
        return enc_feat, lengths, padding_mask, label, modalities,~audioOnlyMask
    def forward_crossmodal(self, x, lengths, label):
        enc_feat, lengths, padding_mask, label, modalities,audio_visualMask = self.getAllModalFeatures(x,lengths,label)
        if self.codec is not None:
            loss_audio = 0
            audios = x["audio"][audio_visualMask]
            if audios.size(0) != 0:
                audios = audios.squeeze(2)
                for modality in self.audioReconstructionModalities:
                    modalityIndex = self.modalities.index(modality)
                    features = enc_feat[modalities==modalityIndex]
                    audio_tokens = self.forward_audio_reconstruction(audios)
                    audio_tokens = audio_tokens[:, : features.size(1) * self.audio_alignment]
                    logits_audio = self.video_classifier(features)
                    logits_audio = logits_audio.float() # converting into float type before the loss calculation
                    logits_audio = logits_audio.unflatten(2, (-1, self.audio_vocab_size))
                    # audio_tokens = audio_tokens.float()
                    # audio_tokens = audio_tokens.unflatten(2, (-1, self.audio_vocab_size))
                    loss_audio += F.cross_entropy(logits_audio.flatten(0, 2),audio_tokens.flatten())
        else:
            loss_audio = None
        loss_ctcMod, ys_hat = self.ctc(enc_feat, lengths["video"], label)
        loss_ctc = loss_ctcMod
        
        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        if self.mtlalpha < 1:
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, enc_feat, padding_mask["video"])
        else:
            pred_pad = None
        loss_att = self.criterion(pred_pad, ys_out_pad)
        loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att
        if self.codec is not None:
            loss = loss + loss_audio * self.audio_weight
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
        return loss, loss_ctc, loss_att, accAll, acc
