package ru.samsu.mj.arnene.mnist;

public class Label {
    private final int label;

    public Label(int label) {
        if(label < 0 || 9 < label) {
            throw new IllegalArgumentException();
        }
        this.label = label;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString() {
        return "Label{" +
            "label=" + label +
            '}';
    }
}
