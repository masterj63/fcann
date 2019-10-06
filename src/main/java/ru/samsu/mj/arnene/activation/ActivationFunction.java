package ru.samsu.mj.arnene.activation;

public enum ActivationFunction {
    GAUSS {
        @Override
        public double getF(double x) {
            return Math.exp(-x * x / 2);
        }

        @Override
        public double getDer(double x) {
            return -1.0 * x * getF(x);
        }
    },
    LINEAR {
        @Override
        public double getF(double x) {
            return x;
        }

        @Override
        public double getDer(double x) {
            return 1.0;
        }
    },
    SIGMA {
        @Override
        public double getF(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public double getDer(double x) {
            {
                double t = 1 + Math.exp(-x);
                double analyticalResult = 1 / (Math.exp(x) * t * t);
//                if (!Double.isNaN(analyticalResult)) {
//                    return analyticalResult;
                return Math.exp(-x) / (1 + Math.exp(-x)) / (1 + Math.exp(-x));
//                }
            }
//            {
//                //1/4 - x^2/16 + x^4/96 - (17 x^6)/11520 + (31 x^8)/161280 - (691 x^10)/29030400 + (5461 x^12)/1916006400 + O(x^13)
//                double xx = 1.0;
//                double res = 0.25;
//
//                xx *= x * x; // 2
//                res -= xx / 16;
//
//                xx *= x * x; // 4
//                res += xx / 96;
//
//                xx *= x * x; // 6
//                res -= xx * 17 / 11520;
//
//                xx *= x * x; // 8
//                res += xx * 31 / 161280;
//
//                xx *= x * x; // 10
//                res -= xx * 691 / 29030400;
//
//                xx *= x * x; // 12
//                res += xx * 5461 / 1916006400;
//
//                if (!Double.isNaN(res)) {
//                    return res;
//                }
//            }
//            {
//
//            }
//            throw new IllegalArgumentException(String.format("failed for %.2f", x));
        }
    },
    HEAVISIDE {
        @Override
        public double getF(double x) {
            return x < 0
                ? 0.0
                : 1.0;
        }

        @Override
        public double getDer(double x) {
            return 0.0;
        }
    },
    TANH {
        @Override
        public double getF(double x) {
            return sinh(x) / cosh(x);
        }

        @Override
        public double getDer(double x) {
            double cosh = cosh(x);
            return 1 / (cosh * cosh);
        }

        private double sinh(double x) {
            return 0.5 * (Math.exp(x) - Math.exp(-x));
        }

        private double cosh(double x) {
            return 0.5 * (Math.exp(x) + Math.exp(-x));
        }
    },
    LOGARITHM {
        @Override
        public double getF(double x) {
            return Math.log(x + Math.sqrt(1.0 + x * x));
        }

        @Override
        public double getDer(double x) {
            return 1.0 / (1.0 + Math.sqrt(x * x));
        }
    },
    RELU {
        @Override
        public double getF(double x) {
            return x < 0
                ? 0.0
                : x;
        }

        @Override
        public double getDer(double x) {
            return x < 0
                ? 0.0
                : 1.0;
        }
    };

    public abstract double getF(double x);

    public abstract double getDer(double x);
}
