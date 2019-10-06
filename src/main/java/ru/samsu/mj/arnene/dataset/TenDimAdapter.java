package ru.samsu.mj.arnene.dataset;

import ru.samsu.mj.arnene.mnist.Label;

public class TenDimAdapter extends DigitAdapter {
    public static final int TEN = 10;

    @Override
    public double[] adaptLabel(Label label) {
        double[] result = new double[TEN];
        result[label.getLabel()] = 1.0;
        return result;
    }

    @Override
    public int getLabelDimension() {
        return TEN;
    }
}