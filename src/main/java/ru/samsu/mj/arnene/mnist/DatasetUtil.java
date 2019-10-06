package ru.samsu.mj.arnene.mnist;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class DatasetUtil {
    private static final long SEED = 0L;

    public static List<Pair> readTrainSet() throws IOException {
        return shufflePairs(readTrainDigits(), readTrainLabels());
    }

    public static List<Pair> readTestSet() throws IOException {
        return shufflePairs(readTestDigits(), readTestLabels());
    }

    private static List<Pair> shufflePairs(List<Digit> digits, List<Label> labels) throws IOException {
        Iterator<Digit> digitIterator = digits.iterator();
        Iterator<Label> labelsIterator = labels.iterator();
        List<Pair> result = new ArrayList<>(digits.size());
        while (digitIterator.hasNext() || labelsIterator.hasNext()) {
            result.add(new Pair(digitIterator.next(), labelsIterator.next()));
        }
        Collections.shuffle(result, new Random(SEED));
        return result;
    }

    private static List<Digit> readTrainDigits() throws IOException {
        return readImages("train-images-idx3-ubyte");
    }

    private static List<Label> readTrainLabels() throws IOException {
        return readLabels("train-labels-idx1-ubyte");
    }

    private static List<Digit> readTestDigits() throws IOException {
        return readImages("t10k-images-idx3-ubyte");
    }

    private static List<Label> readTestLabels() throws IOException {
        return readLabels("t10k-labels-idx1-ubyte");
    }

    private static List<Label> readLabels(String fileName) throws IOException {
        try (InputStream is = DatasetUtil.class.getClassLoader().getResourceAsStream(fileName);
             DataInputStream dis = new DataInputStream(is)) {
            if (dis.readInt() != 0x801) {
                throw new IllegalStateException();
            }

            final int labelsCount = dis.readInt();
            List<Label> result = new ArrayList<>(labelsCount);
            for (int i = 0; i < labelsCount; i++) {
                result.add(new Label(dis.readUnsignedByte()));
            }
            return result;
        }
    }

    private static List<Digit> readImages(String fileName) throws IOException {
        try (InputStream is = DatasetUtil.class.getClassLoader().getResourceAsStream(fileName);
             DataInputStream dis = new DataInputStream(is)) {
            if (dis.readInt() != 0x803) {
                throw new IllegalStateException();
            }

            final int imagesCount = dis.readInt();
            ArrayList<Digit> result = new ArrayList<>(imagesCount);

            final int row = dis.readInt();
            final int col = dis.readInt();
            for (int img = 0; img < imagesCount; img++) {
                int[] canvas = new int[row * col];
                int canvasI = 0;
                for (int i = 0; i < row; i++) {
                    for (int j = 0; j < col; j++) {
                        canvas[canvasI] = dis.readUnsignedByte();
                        canvasI++;
                    }
                }
                result.add(new ByteDigit(canvas));
            }

            return result;
        }
    }

    public static class Pair {
        private final Digit digit;
        private final Label label;

        Pair(Digit digit, Label label) {
            this.digit = digit;
            this.label = label;
        }

        public Digit getDigit() {
            return digit;
        }

        public Label getLabel() {
            return label;
        }
    }
}
