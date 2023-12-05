# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import numpy
import torch

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
            self.encoder["audio"] = self.createEncoder(args["audio"])
            self.encoder["video"] = self.createEncoder(args["video"])
            self.transformer_input_layer["audio"] = args["audio"].transformer_input_layer
            self.transformer_input_layer["video"] = args["video"].transformer_input_layer
            self.a_upsample_ratio["audio"] = args["audio"].a_upsample_ratio
            self.a_upsample_ratio["video"] = args["video"].a_upsample_ratio
            self.proj_decoder["audio"] = None
            self.proj_decoder["video"] = None
            if args["audio"].adim != args["audio"].ddim:
                self.proj_decoder["audio"] = torch.nn.Linear(args["audio"].adim, args["audio"].ddim)
            if args["video"].adim != args["video"].ddim:
                self.proj_decoder["video"] = torch.nn.Linear(args["video"].adim, args["video"].ddim)
            if args["audio"].mtlalpha < 1 or args["video"].mtalpha < 1:
                self.decoder = Decoder(
                    odim=odim,
                    attention_dim=args["video"].ddim,
                    attention_heads=args["video"].dheads,
                    linear_units=args["video"].dunits,
                    num_blocks=args["video"].dlayers,
                    dropout_rate=args["video"].dropout_rate,
                    positional_dropout_rate=args["video"].dropout_rate,
                    self_attention_dropout_rate=args["video"].transformer_attn_dropout_rate,
                    src_attention_dropout_rate=args["video"].transformer_attn_dropout_rate,
                )
            self.fusion = PositionwiseFeedForward(args["fusion"].adim, args["fusion"].hidden_units, args["fusion"].dropout_rate)
            self.criterion = LabelSmoothingLoss(
                self.odim,
                self.ignore_id,
                args["video"].lsm_weight,
                args["video"].transformer_length_normalized_loss,
            )
            self.adim = args["video"].adim
            self.mtlalpha = args["video"].mtlalpha
            if args["video"].mtlalpha > 0.0:
                self.ctc = CTC(
                    odim, args["video"].adim, args["video"].dropout_rate, ctc_type=args["video"].ctc_type, reduce=True
                )
            else:
                self.ctc = None
                
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
    def getModalitites(self, x):
        modality = torch.zeros(x['vid'].size(0), dtype=torch.long, device=x['vid'].device)
        # Determine modality by where video and audio are present (all zeros if not present)
        #check if video is all zeros
        whereVideo = x['vid'].sum(dim=(1,2,3)) != 0
        whereAudio = x['audio'].sum(dim=(1)) != 0
        modality[whereVideo] = 0
        modality[whereAudio] = 1
        modality[whereVideo & whereAudio] = 2
        return modality
    def getCrossModalFeatures(self, x, modality, padding_mask):
        # Initialize combined features tensor
        combined_vid = torch.zeros(modality.size(0), x['vid'].size(1), device=modality.device)
        combined_aud = torch.zeros_like(combined_vid)

        # Process video modality if present
        if 'vid' in x and x['vid'] is not None:
            vid_mask = (modality == 0) | (modality == 2)
            if vid_mask.any():
                vidPaddingMask = None
                if padding_mask is not None:
                    vidPaddingMask = padding_mask[vid_mask]
                xVid = self.encoder["video"](x['vid'][vid_mask], vidPaddingMask)
                combined_vid[vid_mask] = xVid
        
        # Process audio modality if present
        if 'audio' in x and x['audio'] is not None:
            aud_mask = (modality == 1) | (modality == 2)
            if aud_mask.any():
                audPaddingMask = None
                if padding_mask is not None:
                    audPaddingMask = padding_mask[aud_mask]
                xAud = self.encoder["audio"](x['audio'][aud_mask], audPaddingMask)
                combined_aud[aud_mask] = xAud

        # Combine features for modality 2
        combined_mask = modality == 2
        if combined_mask.any():
            x_combined = torch.cat((combined_vid[combined_mask], combined_aud[combined_mask]), dim=2)
            x_combined = self.fusion(x_combined)
            combined_vid[combined_mask] = x_combined
        enc_feat = torch.where(combined_mask[:, None], combined_vid, combined_vid + combined_aud)
        return enc_feat
    def forward_crossmodal(self, x, lengths, label):
        if self.transformer_input_layer == "conv1d":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        modality = self.getModalities(x)
        enc_feat = self.getCrossmodalFeatures(x, modality, padding_mask)
        # ctc loss
        loss_ctc, ys_hat = self.ctc(enc_feat, lengths, label)

        # if self.proj_decoder["audio"] != None:
        #     enc_featAudio = self.proj_decoder["audio"](enc_featAudio)
        # if self.proj_decoder["video"] != None:
        #     enc_featVideo = self.proj_decoder["video"](enc_featVideo)
        
        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        if self.mtlalpha < 1:
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, enc_feat, padding_mask)
        else:
            pred_pad = None
        loss_att = self.criterion(pred_pad, ys_out_pad)
        loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att

        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        return loss, loss_ctc, loss_att, acc
