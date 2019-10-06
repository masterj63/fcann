package ru.samsu.mj.arnene.mnist;

public abstract class Digit {
    public abstract int get(int index);

    public abstract int getPixelCount();

    public void trace(final int rows, final int cols) {
        int i = 0;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                char c = (get(i) < 100)
                    ? ' '
                    : '#';
                System.out.print(c);
                i++;
            }
            System.out.println();
        }
    }
}
