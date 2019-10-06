package ru.samsu.mj.arnene.dataset;

import ru.samsu.mj.arnene.mnist.Digit;

abstract class DigitAdapter implements Adapter {
    @Override
    public int[] adaptDigit(Digit digit) {
        int[] result = new int[digit.getPixelCount()];
        for (int i = 0; i < result.length; i++) {
            result[i] = digit.get(i);
        }
        return result;
    }
}
