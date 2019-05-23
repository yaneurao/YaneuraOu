#include "../types.h"

#include "bitop.h"
#include <iostream>

ymm ymm_zero = ymm(uint8_t(0));
ymm ymm_one = ymm(uint8_t(1));

void* aligned_malloc(size_t size, size_t align)
{
	void* p = _mm_malloc(size, align);
	if (p == nullptr)
	{
		std::cout << "info string can't allocate memory. sise = " << size << std::endl;
		exit(1);
	}
	return p;
}
