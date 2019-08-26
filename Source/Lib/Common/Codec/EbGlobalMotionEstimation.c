
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
    uint8_t num_of_ref_pic_to_search = AOMMIN(picture_control_set_ptr->ref_list0_count, 1);

    // Ref Picture Loop
    for (uint32_t ref_pic_index = 0; ref_pic_index < num_of_ref_pic_to_search;
         ++ref_pic_index)
    {
        EbPaReferenceObject *referenceObject;

        if (context_ptr->me_alt_ref == EB_TRUE)
            referenceObject = (EbPaReferenceObject *)context_ptr->alt_ref_reference_ptr;
        else
            referenceObject = (EbPaReferenceObject *)picture_control_set_ptr
                    ->ref_pa_pic_ptr_array[REF_LIST_0][ref_pic_index]->object_ptr;


        EbPictureBufferDesc *ref_picture_ptr = (EbPictureBufferDesc*)referenceObject->input_padded_picture_ptr;

        compute_global_motion(input_picture_ptr, ref_picture_ptr, &picture_control_set_ptr->global_motion_estimation);
    }
}


/* TODO: fix this */
static INLINE int convert_to_trans_prec(int allow_hp, int coor) {
    if (allow_hp)
        return ROUND_POWER_OF_TWO_SIGNED(coor, WARPEDMODEL_PREC_BITS - 3);
    else
        return ROUND_POWER_OF_TWO_SIGNED(coor, WARPEDMODEL_PREC_BITS - 2) * 2;
}


void compute_global_motion(EbPictureBufferDesc *input_pic, EbPictureBufferDesc *ref_pic,
                           EbWarpedMotionParams *bestWarpedMotion)
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
    unsigned char *frm_buffer = input_pic->buffer_y + input_pic->origin_x + input_pic->origin_y * input_pic->stride_y;
    unsigned char *ref_buffer = ref_pic->buffer_y + ref_pic->origin_x + ref_pic->origin_y * ref_pic->stride_y;
    // TODO: handle the > 8 bits cases.

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
                        ref_buffer, ref_pic->stride_y, ref_pic->bit_depth,
                        gm_estimation_type, inliers_by_motion, params_by_motion,
                        RANSAC_NUM_MOTIONS);

            for (unsigned i = 0; i < RANSAC_NUM_MOTIONS; ++i) {
                if (inliers_by_motion[i] == 0) continue;

                params_this_motion = params_by_motion[i].params;
                av1_convert_model_to_params(params_this_motion, &tmp_wm_params);

                if (tmp_wm_params.wmtype != IDENTITY) {
                    const int64_t warp_error = av1_refine_integerized_param(
                                &tmp_wm_params, tmp_wm_params.wmtype, ref_pic->bit_depth > 8 /* High bitrate */,
                                ref_pic->bit_depth,
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
                if (!get_shear_params(&global_motion))
                    global_motion = default_warp_params;

            if (global_motion.wmtype == TRANSLATION) {
                global_motion.wmmat[0] =
                        convert_to_trans_prec(0, // TODO: check this allow_high_precision_mv
                                              global_motion.wmmat[0]) *
                        GM_TRANS_ONLY_DECODE_FACTOR;
                global_motion.wmmat[1] =
                        convert_to_trans_prec(0, // TODO: check this allow_high_precision_mv
                                              global_motion.wmmat[1]) *
                        GM_TRANS_ONLY_DECODE_FACTOR;
            }

            if (global_motion.wmtype == IDENTITY)
                continue;

            const int64_t ref_frame_error = av1_frame_error(
                        ref_pic->bit_depth > 8 /* High bitrate */, ref_pic->bit_depth, ref_buffer, ref_pic->stride_y, frm_buffer,
                        input_pic->width, input_pic->height, input_pic->stride_y);

            if (ref_frame_error == 0)
                continue;

            // If the best error advantage found doesn't meet the threshold for
            // this motion type, revert to IDENTITY.
            if (!av1_is_enough_erroradvantage(
                        (double)best_warp_error / ref_frame_error,
                        gm_get_params_cost(&global_motion, ref_params,
                                           0 /* TODO: check this allow_high_precision_mv */),
                        GM_ERRORADV_TR_0 /* TODO: check error advantage */)) {
                global_motion = default_warp_params;
            }
            if (global_motion.wmtype != IDENTITY) {
                break;
            }
        }
    }

    *bestWarpedMotion = global_motion;

    for (int m = 0; m < RANSAC_NUM_MOTIONS; m++) {
        free(params_by_motion[m].inliers);
    }
}