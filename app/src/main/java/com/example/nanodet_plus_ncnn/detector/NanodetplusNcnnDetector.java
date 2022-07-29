package com.example.nanodet_plus_ncnn.detector;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

public class NanodetplusNcnnDetector {

    public class BoxInfo {
        public float x1;
        public float y1;
        public float x2;
        public float y2;
//        public float w;
//        public float h;
        public float score;
        public int label;
    }

    private final String[] COCO_LABELS = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };
    private final String[] QR_LABELS = {"QR"};
    public String[] LABELS;

    private final String MODEL_NANODET_M_COCO = "nanodet-plus-coco-416";
    private final String MODEL_NANODETPLUS_QR = "nanodet-plus-qr-416";

    public String MODEL_FILE;
    public Integer NUM_CLASSES;

    public NanodetplusNcnnDetector(Activity activity, String modelName) {
        setModelFile(modelName);
        init(activity.getAssets(), getModelFile());
        Log.i("ncnn:","initial model success!");
    }

    public String getLabel(int labelId) {return this.LABELS[labelId];}

    public String getModelFile() {
        return this.MODEL_FILE;
    }

    public Integer getNumClasses(){return this.NUM_CLASSES;}


    public void setModelFile(String modelFile){
        switch (modelFile) {
            case "nanodet-plus-coco-416":
                MODEL_FILE = this.MODEL_NANODET_M_COCO;
                LABELS = this.COCO_LABELS;
                NUM_CLASSES = LABELS.length;
                break;
            case "nanodet-plus-qr-416":
                MODEL_FILE = this.MODEL_NANODETPLUS_QR;
                LABELS = this.QR_LABELS;
                NUM_CLASSES = LABELS.length;
                break;
            default:
                Log.i("ncnn:", "cannot find target model: "+modelFile);
        }
    }

    public native boolean init(AssetManager assetManager, String modelName);

//    public native NanodetplusNcnnDetector.BoxInfo[] detect(
//            Bitmap bitmap,
//            int num_classes,
//            int rotation,
//            int crop_w,
//            int crop_h,
//            int preview_w,
//            int preview_h);
    public native NanodetplusNcnnDetector.BoxInfo[] detect(
            Bitmap bitmap,
            int num_classes);

    static {
        System.loadLibrary("nanodet_plus_ncnn");
    }
}
