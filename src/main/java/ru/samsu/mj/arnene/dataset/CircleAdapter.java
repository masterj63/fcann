package ru.samsu.mj.arnene.dataset;

import ru.samsu.mj.arnene.mnist.Label;

public class CircleAdapter extends DigitAdapter {
    @Override
    public double[] adaptLabel(Label label) {
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

    @Override
    public int getLabelDimension() {
        return 1;
    }
}
