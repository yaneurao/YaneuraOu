#include "math.h"

#include <cmath>

double Math::sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

double Math::dsigmoid(double x) {
  return sigmoid(x) * (1.0 - sigmoid(x));
}
