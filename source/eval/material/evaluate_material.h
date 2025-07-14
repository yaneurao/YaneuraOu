#ifndef EVALUATE_MATERIAL_H_INCLUDED
#define EVALUATE_MATERIAL_H_INCLUDED

#include "../../config.h"
#if defined(EVAL_MATERIAL)

#include "../../types.h"

namespace YaneuraOu {
namespace Eval {

class MaterialEvaluator: public IEvaluator {
   public:
    virtual void init() override;
    virtual Value evaluate(const Position& pos) override;
};

} // namespace Eval
} // namespace YaneuraOu

#endif

#endif  // EVALUATE_MATERIAL_H_INCLUDED
