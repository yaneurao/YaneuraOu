#include "../shogi.h"

#if defined (EVAL_NO_USE)
namespace Eval
{
	void init() {}
	void load_eval() {}
	void print_eval_stat(Position& pos) {}
	Value evaluate(const Position& pos) { return VALUE_NONE; }
	Value compute_eval(const Position& pos) { return VALUE_NONE; }
}
#endif
