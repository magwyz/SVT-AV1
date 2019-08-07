
#ifndef EbGlobalMotionEstimationCost_h
#define EbGlobalMotionEstimationCost_h

#include "EbDefinitions.h"


int gm_get_params_cost(const EbWarpedMotionParams *gm,
                       const EbWarpedMotionParams *ref_gm, int allow_hp);

#endif
