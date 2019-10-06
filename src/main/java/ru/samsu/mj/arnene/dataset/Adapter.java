package ru.samsu.mj.arnene.dataset;

import ru.samsu.mj.arnene.mnist.Digit;
import ru.samsu.mj.arnene.mnist.Label;

public interface Adapter {
    int[] adaptDigit(Digit digit);

    double[] adaptLabel(Label label);

    int getLabelDimension();
}
