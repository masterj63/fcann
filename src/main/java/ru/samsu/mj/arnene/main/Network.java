package ru.samsu.mj.arnene.main;

import ru.samsu.mj.arnene.activation.ActivationFunction;

import java.io.Serializable;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

public class Network implements Serializable {
    private static final long SEED = 0L;
    private static final Random RANDOM = new Random(SEED);

    private final int inputLayerSize;
    private final int hiddenLayerSize;
    private final int outputLayerSize;
    private final ActivationFunction activationFunction;

    private final double[][] wnH;
    private final double[][] wHM;

    private String label;

    private Network(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, ActivationFunction activationFunction) {
        this.inputLayerSize = inputLayerSize;
        this.hiddenLayerSize = hiddenLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.activationFunction = activationFunction;

        this.wnH = initializeWeights(1 + inputLayerSize, hiddenLayerSize);
        this.wHM = initializeWeights(1 + hiddenLayerSize, outputLayerSize);
    }

    private static double[][] initializeWeights(int dim1, int dim2) {
        double[][] a = new double[dim1][dim2];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                a[i][j] = random(0.5 / a.length);
            }
        }
        return a;
    }

    private static double random(double absBound) {
        return (RANDOM.nextDouble() - 0.5) * absBound;
    }

    private boolean isWhmNanny() {
        return Stream.of(wHM)
            .flatMapToDouble(DoubleStream::of)
            .anyMatch(Double::isNaN);
    }

    private boolean isWhnNanny() {
        return Stream.of(wnH)
            .flatMapToDouble(DoubleStream::of)
            .anyMatch(Double::isNaN);
    }

    public PropagationResult propagate(int[] inputs) {
        if (inputs.length != inputLayerSize) {
            throw new IllegalArgumentException();
        }

        double[] preHidden = new double[hiddenLayerSize];
        double[] hidden;
        {
            for (int j = 0; j <= inputLayerSize; j++) {
                double input = (j == 0)
                    ? -1.0 // mix "-1" in
                    : inputs[j - 1];
                for (int h = 0; h < hiddenLayerSize; h++) {
                    preHidden[h] += wnH[j][h] * input;
                }
            }
            hidden = new double[1 + hiddenLayerSize];
            hidden[0] = -1.0;
            for (int i = 1; i <= hiddenLayerSize; i++) {
                hidden[i] = activationFunction.getF(preHidden[i - 1]);
            }
        }

        double[] preOutput = new double[outputLayerSize];
        double[] output;
        {
            for (int h = 0; h <= hiddenLayerSize; h++) {
                for (int m = 0; m < outputLayerSize; m++) {
                    preOutput[m] += wHM[h][m] * hidden[h];
                }
            }
            output = DoubleStream.of(preOutput)
                .map(activationFunction::getF)
                .toArray();
        }

        return new PropagationResult(preHidden, hidden, preOutput, output);
    }

    public double[] getHiddenError(double[] em, double[] outputPreactivation) {
        double[] result = new double[hiddenLayerSize];
        for (int h = 0; h < hiddenLayerSize; h++) {
            for (int m = 1; m <= outputLayerSize; m++) {
                double actFunArg = outputPreactivation[m - 1];
                double der = activationFunction.getDer(actFunArg);
                result[h] += em[m - 1] * der * wHM[h][m - 1];
            }
        }
        return result;
    }

    public void tuneWhm(double[] emOutputError, double[] outputPreactivation, double[] hiddenActivation, double eta) {
        for (int h = 0; h <= hiddenLayerSize; h++) {
            for (int m = 0; m < outputLayerSize; m++) {
                wHM[h][m] -= eta
                    * emOutputError[m]
                    * activationFunction.getDer(outputPreactivation[m])
                    * hiddenActivation[h];
            }
        }
    }

    public void tuneWnh(double[] ehHiddenError, double[] hiddenPreactivation, int[] input, double eta) {
        for (int j = 0; j < inputLayerSize; j++) {
            for (int h = 0; h < hiddenLayerSize; h++) {
                double v = eta
                    * ehHiddenError[h]
                    * activationFunction.getDer(hiddenPreactivation[h])
                    * input[j];
//                if (Double.isNaN(v)) {
//                    System.out.println("is nan");
//                }
                wnH[j][h] -= v;
            }
        }
    }

    private double getAvg(double[][] a) {
        return Stream.of(a)
            .flatMapToDouble(DoubleStream::of)
            .average()
            .getAsDouble();
    }

    public double getAvgNh() {
        return getAvg(wnH);
    }

    public double getAvgHm() {
        return getAvg(wHM);
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getActivationFunctionName() {
        return activationFunction.name().toLowerCase();
    }

    public static class PropagationResult {
        // N and (1+N) dimensions (plus one for "-1" node):
        public final double[] hiddenPreactivation;
        public final double[] outputPreactivation;
        // same dimension:
        public double[] hiddenActivation;
        public double[] outputActivation;

        public PropagationResult(double[] hiddenPreactivation, double[] hiddenActivation,
                                 double[] outputPreactivation, double[] outputActivation) {
            this.hiddenPreactivation = hiddenPreactivation;
            this.hiddenActivation = hiddenActivation;
            this.outputPreactivation = outputPreactivation;
            this.outputActivation = outputActivation;
        }
    }

    public static class Builder {
        private int inputLayerSize;
        private int hiddenLayerSize;
        private int outputLayerSize;
        private ActivationFunction activationFunction;

        public Network build() {
            return new Network(inputLayerSize, hiddenLayerSize, outputLayerSize, activationFunction);
        }

        public Builder setInputLayerSize(int inputLayerSize) {
            this.inputLayerSize = inputLayerSize;
            return this;
        }

        public Builder setHiddenLayerSize(int hiddenLayerSize) {
            this.hiddenLayerSize = hiddenLayerSize;
            return this;
        }

        public Builder setOutputLayerSize(int outputLayerSize) {
            this.outputLayerSize = outputLayerSize;
            return this;
        }

        public Builder setActivationFunction(ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }
    }
}