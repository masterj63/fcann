package ru.samsu.mj.arnene;

import ru.samsu.mj.arnene.activation.ActivationFunction;
import ru.samsu.mj.arnene.main.Network;
import ru.samsu.mj.arnene.main.Network.PropagationResult;
import ru.samsu.mj.arnene.mnist.BooleanDigit;
import ru.samsu.mj.arnene.mnist.DatasetUtil;
import ru.samsu.mj.arnene.mnist.Digit;
import ru.samsu.mj.arnene.mnist.Label;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.DecimalFormat;
import java.util.Date;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class App {
    private static final double ETA = 0.004;
    private static final DecimalFormat FORMATTER = new DecimalFormat("0.###E0");
    private static final ActivationFunction ACTIVATION_FUNCTION = ActivationFunction.SIGMA;
    private static final int HIDDEN_LAYER_SIZE = 3;
    private static final File MODELS_DIR = new File("models");
    private static final String ANN_FILE_NAME = String.format("ann.%s.%d",
        ACTIVATION_FUNCTION.name().toLowerCase(), HIDDEN_LAYER_SIZE);

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        App app = new App();
        app.run();
    }

    private static double[] subtract(double[] vec0, double[] vec1) {
        if (vec0.length != vec1.length) {
            throw new IllegalArgumentException();
        }
        double[] result = new double[vec0.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = vec0[i] - vec1[i];
        }
        return result;
    }

    private static double sumSquare(double[] vec) {
        return DoubleStream.of(vec).map(x -> x * x).sum();
    }

    private Network trainNetwork(List<BooleanDigit> digits, List<Label> labels) {
        // p108.1
        Network network = new Network.Builder()
            .setActivationFunction(ACTIVATION_FUNCTION)
            .setInputLayerSize(digits.get(0).getPixelCount())
            .setHiddenLayerSize(HIDDEN_LAYER_SIZE)
            .setOutputLayerSize(1)
            .build();

        double q = 0.0;
        // p108.2
        for (int i = 0; i < digits.size(); i++) {
            boolean lastIteration = ((1 + i) == digits.size());

            // p108.3
            int[] xi = adaptDigit(digits.get(i));
            // p108.4 // forward propagation
            PropagationResult propagation = network.propagate(xi);
            double[] aa = propagation.outputActivation;
            double[] yy = adaptLabel(labels.get(i));
            double[] emOutputError = subtract(aa, yy);
            double qi = 0.5 * sumSquare(emOutputError);
            // p108.5 backward propagation
            double[] ehHiddenError = network.getHiddenError(emOutputError, propagation.outputPreactivation);
            // p108.6 gradient computation
            network.tuneWhm(emOutputError, propagation.outputPreactivation, propagation.hiddenActivation, ETA);
            network.tuneWnh(ehHiddenError, propagation.hiddenPreactivation, xi, ETA);
            // p108.7
            double newQ = 1.0 * (digits.size() - 1) / digits.size() * q + 1.0 / digits.size() * qi;
            q = newQ;
            // p108.8
            if (i > 30_000 && Math.min(q, newQ) / Math.max(q, newQ) > 0.986) {
                lastIteration = true;
            }

            if (i % 1_000 == 0 || lastIteration) {
                String message = String.format(
                    "%s %.0f%% (%d): q=%s, avg_x_i=%s, out_err=%s, hid_err=%s, nw_avg_nh=%s, nw_avg_hm=%s",
                    new Date(),
                    100.0 * i / digits.size(),
                    i,
                    FORMATTER.format(q),
                    FORMATTER.format(IntStream.of(xi).average().getAsDouble()),
                    FORMATTER.format(DoubleStream.of(emOutputError).average().getAsDouble()),
                    FORMATTER.format(DoubleStream.of(ehHiddenError).average().getAsDouble()),
                    FORMATTER.format(network.getAvgNh()),
                    FORMATTER.format(network.getAvgHm())
                );
                network.setLabel(message);
                System.out.println(message);
            }
            if (lastIteration) {
                break;
            }
        }

        return network;
    }

    private void run() throws IOException, ClassNotFoundException {
        Network network;
        if (!MODELS_DIR.exists()) {
            MODELS_DIR.mkdir();
        }
        final File annFile = new File(MODELS_DIR, ANN_FILE_NAME);
        if (annFile.exists()) {
            try (FileInputStream fileInputStream = new FileInputStream(annFile);
                 ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream)) {
                network = (Network) objectInputStream.readObject();
            }
            System.out.printf("deserialized %s %s%n", network.getActivationFunctionName(), network.getLabel());
        } else {
            List<BooleanDigit> digits = DatasetUtil.readTrainDigits();
            List<Label> labels = DatasetUtil.readTrainLabels();
            network = trainNetwork(digits, labels);
            try (FileOutputStream fos = new FileOutputStream(annFile);
                 ObjectOutputStream oos = new ObjectOutputStream(fos)) {
                oos.writeObject(network);
            }
            System.out.printf("serialized %s %s%n", network.getActivationFunctionName(), network.getLabel());
        }

        List<BooleanDigit> digits = DatasetUtil.readTestDigits();
        List<Label> labels = DatasetUtil.readTestLabels();

        if (digits.size() != labels.size()) {
            throw new IllegalStateException();
        }
        double sum = 0.0;
        int N = digits.size();
        for (int i = 0; i < N; i++) {
            double actual = network.propagate(adaptDigit(digits.get(i))).outputActivation[0];
            double expected = adaptLabel(labels.get(i))[0];
            double diff = actual - expected;
//            double diff2 = diff * diff;
            sum += Math.abs(diff);
        }
        System.out.printf("avg error on test set: %.4f \n", 1.0 * sum / N);
//        for (int i = 0; i < 10; i++) {
//            Digit digit = digits.get(i);
//            Label label = labels.get(i);
//
//            digit.trace(28, 28); // TODO
//            double[] output = network.propagate(adaptDigit(digit)).outputActivation;
//            System.out.println(Arrays.toString(output));
//            System.out.println(label);
//            System.out.println("### " + i);
//        }
    }

    private int[] adaptDigit(Digit digit) {
        int[] result = new int[digit.getPixelCount()];
        for (int i = 0; i < result.length; i++) {
            result[i] = digit.get(i);
        }
        return result;
    }

    private double[] adaptLabel(Label label) {
        switch (label.getLabel()) {
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 7:
                return new double[]{0.0};
            case 0:
            case 6:
            case 9:
                return new double[]{1.0};
            case 8:
                return new double[]{2.0};
            default:
                throw new IllegalArgumentException(String.valueOf(label));
        }
    }
}
