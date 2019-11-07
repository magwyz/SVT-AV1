/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

/*
* Copyright (c) 2016, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at www.aomedia.org/license/software. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at www.aomedia.org/license/patent.
*/

#include <stdlib.h>

#include "EbGlobalMotionEstimation.h"
#include "EbGlobalMotionEstimationCost.h"
#include "EbReferenceObject.h"

#include "global_motion.h"
#include "corner_detect.h"



void global_motion_estimation(PictureParentControlSet *picture_control_set_ptr,
                              MeContext *context_ptr,
                              EbPictureBufferDesc *input_picture_ptr)
{
    uint32_t numOfListToSearch = (picture_control_set_ptr->slice_type == P_SLICE)
        ? (uint32_t)REF_LIST_0 : (uint32_t)REF_LIST_1;

    for (uint32_t listIndex = REF_LIST_0; listIndex <= numOfListToSearch; ++listIndex) {

        uint32_t num_of_ref_pic_to_search;
        if (context_ptr->me_alt_ref == EB_TRUE)
            num_of_ref_pic_to_search = 1;
        else
            num_of_ref_pic_to_search = picture_control_set_ptr->slice_type == P_SLICE
                ? picture_control_set_ptr->ref_list0_count
                : listIndex == REF_LIST_0
                    ? picture_control_set_ptr->ref_list0_count
                    : picture_control_set_ptr->ref_list1_count;

        // Limit the global motion search to the first frame types of ref lists
        num_of_ref_pic_to_search = MIN(num_of_ref_pic_to_search, 1);

        // Ref Picture Loop
        for (uint32_t ref_pic_index = 0; ref_pic_index < num_of_ref_pic_to_search;
             ++ref_pic_index)
        {
            EbPaReferenceObject *referenceObject;

            if (context_ptr->me_alt_ref == EB_TRUE)
                referenceObject = (EbPaReferenceObject *)context_ptr->alt_ref_reference_ptr;
            else
                referenceObject = (EbPaReferenceObject *)picture_control_set_ptr
                        ->ref_pa_pic_ptr_array[listIndex][ref_pic_index]->object_ptr;


            EbPictureBufferDesc *ref_picture_ptr = (EbPictureBufferDesc*)referenceObject->input_padded_picture_ptr;

            compute_global_motion(input_picture_ptr, ref_picture_ptr,
                &picture_control_set_ptr->global_motion_estimation[listIndex][ref_pic_index],
                picture_control_set_ptr->frm_hdr.allow_high_precision_mv);
        }
    }
}


static INLINE int convert_to_trans_prec(int allow_hp, int coor) {
    if (allow_hp)
        return ROUND_POWER_OF_TWO_SIGNED(coor, WARPEDMODEL_PREC_BITS - 3);
    else
        return ROUND_POWER_OF_TWO_SIGNED(coor, WARPEDMODEL_PREC_BITS - 2) * 2;
}


static unsigned char *av1_downconvert_frame(EbPictureBufferDesc *input_pic) {
    int i, j;
    uint16_t *orig_buf = (uint16_t *)input_pic->buffer_y;
    uint8_t *buf_8bit = malloc(input_pic->stride_y * input_pic->height);
    if (buf_8bit == NULL)
        return NULL;

    for (i = 0; i < input_pic->height; ++i) {
        for (j = 0; j < input_pic->width; ++j) {
            buf_8bit[i * input_pic->stride_y + j] =
                orig_buf[i * input_pic->stride_y + j] >> (input_pic->bit_depth - 8);
        }
    }
    return buf_8bit;
}


void compute_global_motion(EbPictureBufferDesc *input_pic, EbPictureBufferDesc *ref_pic,
                           EbWarpedMotionParams *bestWarpedMotion, int allow_high_precision_mv)
{
    MotionModel params_by_motion[RANSAC_NUM_MOTIONS];
    for (int m = 0; m < RANSAC_NUM_MOTIONS; m++) {
        memset(&params_by_motion[m], 0, sizeof(params_by_motion[m]));
        params_by_motion[m].inliers =
                malloc(sizeof(*(params_by_motion[m].inliers)) * 2 * MAX_CORNERS);
    }

    const double *params_this_motion;
    int inliers_by_motion[RANSAC_NUM_MOTIONS];
    EbWarpedMotionParams tmp_wm_params;
    // clang-format off
    static const double kIdentityParams[MAX_PARAMDIM - 1] = {
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    };
    // clang-format on
    int frm_corners[2 * MAX_CORNERS];
    unsigned char *frm_buffer = input_pic->bit_depth > EB_8BIT ?
                av1_downconvert_frame(input_pic)
                : input_pic->buffer_y + input_pic->origin_x + input_pic->origin_y * input_pic->stride_y;
    if (frm_buffer == NULL)
        return;
    unsigned char *ref_buffer = ref_pic->bit_depth > EB_8BIT ?
                av1_downconvert_frame(ref_pic)
                : ref_pic->buffer_y + ref_pic->origin_x + ref_pic->origin_y * ref_pic->stride_y;
    if (ref_buffer == NULL)
        return;

    EbWarpedMotionParams global_motion = default_warp_params;

    // TODO: check ref_params
    const EbWarpedMotionParams *ref_params = &default_warp_params;

    {
        // compute interest points using FAST features
        int num_frm_corners = av1_fast_corner_detect(
            frm_buffer, input_pic->width, input_pic->height,
            input_pic->stride_y, frm_corners, MAX_CORNERS);

        TransformationType model;
        #define GLOBAL_TRANS_TYPES_ENC 3

        const GlobalMotionEstimationType gm_estimation_type = GLOBAL_MOTION_FEATURE_BASED;
        for (model = ROTZOOM; model <= GLOBAL_TRANS_TYPES_ENC; ++model) {
            int64_t best_warp_error = INT64_MAX;
            // Initially set all params to identity.
            for (unsigned i = 0; i < RANSAC_NUM_MOTIONS; ++i) {
                memcpy(params_by_motion[i].params, kIdentityParams,
                       (MAX_PARAMDIM - 1) * sizeof(*(params_by_motion[i].params)));
            }

            av1_compute_global_motion(
                        model, frm_buffer, input_pic->width, input_pic->height,
                        input_pic->stride_y, frm_corners, num_frm_corners,
                        ref_buffer, ref_pic->stride_y, EB_8BIT,
                        gm_estimation_type, inliers_by_motion, params_by_motion,
                        RANSAC_NUM_MOTIONS);

            for (unsigned i = 0; i < RANSAC_NUM_MOTIONS; ++i) {
                if (inliers_by_motion[i] == 0) continue;

                params_this_motion = params_by_motion[i].params;
                av1_convert_model_to_params(params_this_motion, &tmp_wm_params);

                if (tmp_wm_params.wmtype != IDENTITY) {
                    const int64_t warp_error = av1_refine_integerized_param(
                                &tmp_wm_params, tmp_wm_params.wmtype, EB_FALSE, EB_8BIT,
                                ref_buffer, ref_pic->width, ref_pic->height, ref_pic->stride_y,
                                frm_buffer, input_pic->width, input_pic->height, input_pic->stride_y, 5,
                                best_warp_error);
                    if (warp_error < best_warp_error) {
                        best_warp_error = warp_error;
                        // Save the wm_params modified by
                        // av1_refine_integerized_param() rather than motion index to
                        // avoid rerunning refine() below.
                        memcpy(&global_motion, &tmp_wm_params,
                               sizeof(EbWarpedMotionParams));
                    }
                }
            }
            if (global_motion.wmtype <= AFFINE)
                if (!eb_get_shear_params(&global_motion))
                    global_motion = default_warp_params;

            if (global_motion.wmtype == TRANSLATION) {
                global_motion.wmmat[0] =
                        convert_to_trans_prec(allow_high_precision_mv,
                                              global_motion.wmmat[0]) *
                        GM_TRANS_ONLY_DECODE_FACTOR;
                global_motion.wmmat[1] =
                        convert_to_trans_prec(allow_high_precision_mv,
                                              global_motion.wmmat[1]) *
                        GM_TRANS_ONLY_DECODE_FACTOR;
            }

            if (global_motion.wmtype == IDENTITY)
                continue;

            const int64_t ref_frame_error = eb_av1_frame_error(
                        EB_FALSE, EB_8BIT, ref_buffer, ref_pic->stride_y, frm_buffer,
                        input_pic->width, input_pic->height, input_pic->stride_y);

            if (ref_frame_error == 0)
                continue;

            // If the best error advantage found doesn't meet the threshold for
            // this motion type, revert to IDENTITY.
            if (!av1_is_enough_erroradvantage(
                        (double)best_warp_error / ref_frame_error,
                        gm_get_params_cost(&global_motion, ref_params,
                                           allow_high_precision_mv),
                        GM_ERRORADV_TR_0 /* TODO: check error advantage */)) {
                global_motion = default_warp_params;
            }
            if (global_motion.wmtype != IDENTITY) {
                break;
            }
        }
    }

    *bestWarpedMotion = global_motion;

    if (input_pic->bit_depth > EB_8BIT)
        free(frm_buffer);
    if (ref_pic->bit_depth > EB_8BIT)
        free(ref_buffer);

    for (int m = 0; m < RANSAC_NUM_MOTIONS; m++) {
        free(params_by_motion[m].inliers);
    }
}
