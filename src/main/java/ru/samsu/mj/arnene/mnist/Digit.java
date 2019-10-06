package ru.samsu.mj.arnene.mnist;

public abstract class Digit {
    public abstract int get(int index);

    public abstract int getPixelCount();

    public int getRows() {
        return 28;
    }

    public int getCols() {
        return 28;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder()
            .append(getClass().getSimpleName())
            .append("{\n");
        int N = getPixelCount();
        for (int i = 0; i < N; i++) {
            char c = (get(i) < 100)
                ? ' '
                : '#';
            stringBuilder.append(c);
            if ((1 + i) % getCols() == 0) {
                stringBuilder.append("\n");
            }
        }
        stringBuilder.append("}");
        return stringBuilder.toString();
    }
}
