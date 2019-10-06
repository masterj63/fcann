package ru.samsu.mj.arnene.mnist;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class DatasetUtil {
//    private static final int DEBUG = 10_000; // TODO
    private static final int DEBUG = Integer.MAX_VALUE; // TODO

    public static List<Digit> readTrainDigits() throws IOException {
        return readImages("train-images-idx3-ubyte");
    }

    public static List<Digit> readTestDigits() throws IOException {
        return readImages("t10k-images-idx3-ubyte");
    }

    public static List<Label> readTrainLabels() throws IOException {
        return readLabels("train-labels-idx1-ubyte");
    }

    public static List<Label> readTestLabels() throws IOException {
        return readLabels("t10k-labels-idx1-ubyte");
    }

    private static List<Label> readLabels(String fileName) throws IOException {
        try (InputStream is = DatasetUtil.class.getClassLoader().getResourceAsStream(fileName);
             DataInputStream dis = new DataInputStream(is)) {
            if (dis.readInt() != 0x801) {
                throw new IllegalStateException();
            }

            final int labelsCount = Math.min(dis.readInt(), DEBUG);
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

            final int imagesCount = Math.min(dis.readInt(), DEBUG);
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
}
