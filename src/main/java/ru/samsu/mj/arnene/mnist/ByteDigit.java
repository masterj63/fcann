package ru.samsu.mj.arnene.mnist;

public class ByteDigit extends Digit {
    private final int[] canvas;

    public ByteDigit(int[] canvas) {
        this.canvas = canvas;
    }

    @Override
    public int get(int index) {
        return canvas[index];
    }

    @Override
    public int getPixelCount() {
        return canvas.length;
    }
}
