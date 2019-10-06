package ru.samsu.mj.arnene.mnist;

public class BooleanDigit extends Digit {
    private final boolean[] canvas;
    private final int scale;

    public BooleanDigit(int[] intCanvas, int scale) {
        this.scale = scale;
        this.canvas = new boolean[intCanvas.length];
        for (int i = 0; i < canvas.length; i++) {
            canvas[i] = (intCanvas[i] != 0);
        }
    }

    @Override
    public int get(int index) {
        return canvas[index]
            ? scale
            : 0;
    }

    @Override
    public int getPixelCount() {
        return canvas.length;
    }
}
