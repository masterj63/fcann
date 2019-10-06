package ru.samsu.mj.arnene.dataset;

import ru.samsu.mj.arnene.mnist.Label;

public class IdAdapter extends DigitAdapter {
    @Override
    public double[] adaptLabel(Label label) {
        return new double[]{label.getLabel()};
    }

    @Override
    public int getLabelDimension() {
        return 1;
    }
}
