
#ifndef EbGlobalMotionEstimation_h
#define EbGlobalMotionEstimation_h

#include "EbPictureBufferDesc.h"
#include "EbMotionEstimationContext.h"
#include "EbMotionEstimationProcess.h"


void global_motion_estimation(PictureParentControlSet *picture_control_set_ptr,
                              MeContext *context_ptr,
                              EbPictureBufferDesc *input_picture_ptr);
void compute_global_motion(EbPictureBufferDesc *input_pic, EbPictureBufferDesc *ref_pic,
                           EbWarpedMotionParams *bestWarpedMotion, int allow_high_precision_mv);


#endif // EbGlobalMotionEstimation_h
