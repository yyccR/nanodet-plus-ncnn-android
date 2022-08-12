package com.example.nanodet_plus_ncnn.analysis;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import com.example.nanodet_plus_ncnn.detector.NanodetplusNcnnDetector;
import com.example.nanodet_plus_ncnn.utils.ImageProcess;
import com.example.nanodet_plus_ncnn.utils.Recognition;

import java.util.ArrayList;

import io.reactivex.rxjava3.android.schedulers.AndroidSchedulers;
import io.reactivex.rxjava3.core.Observable;
import io.reactivex.rxjava3.core.ObservableEmitter;
import io.reactivex.rxjava3.schedulers.Schedulers;

public class FullImageAnalyse implements ImageAnalysis.Analyzer {

    public static class Result{

        public Result(long costTime, Bitmap bitmap) {
            this.costTime = costTime;
            this.bitmap = bitmap;
        }
        long costTime;
        Bitmap bitmap;
    }
    int flap = 0;

    ImageView boxLabelCanvas;
    PreviewView previewView;
    int rotation;
    private TextView inferenceTimeTextView;
    private TextView frameSizeTextView;
    ImageProcess imageProcess;
    private NanodetplusNcnnDetector nanodetplusNcnnDetector;

    public FullImageAnalyse(Context context,
                            PreviewView previewView,
                            ImageView boxLabelCanvas,
                            int rotation,
                            TextView inferenceTimeTextView,
                            TextView frameSizeTextView,
                            NanodetplusNcnnDetector nanodetplusNcnnDetector) {
        this.previewView = previewView;
        this.boxLabelCanvas = boxLabelCanvas;
        this.rotation = rotation;
        this.inferenceTimeTextView = inferenceTimeTextView;
        this.frameSizeTextView = frameSizeTextView;
        this.imageProcess = new ImageProcess();
        this.nanodetplusNcnnDetector = nanodetplusNcnnDetector;
    }

    @Override
    public void analyze(@NonNull ImageProxy image) {
//        if(flap > 10){
//            return;
//        }else{
//            flap++;
//        }
        int previewHeight = previewView.getHeight();
        int previewWidth = previewView.getWidth();
        Log.i("preview hw", previewHeight+"/"+previewWidth);

        // 这里Observable将image analyse的逻辑放到子线程计算, 渲染UI的时候再拿回来对应的数据, 避免前端UI卡顿
        Observable.create( (ObservableEmitter<Result> emitter) -> {
            long start = System.currentTimeMillis();

//            byte[][] yuvBytes = new byte[3][];
//            ImageProxy.PlaneProxy[] planes = image.getPlanes();
//
            int imageHeight = image.getHeight();
            int imagewWidth = image.getWidth();
//            int rotate_w = rotation % 180 == 0 ? imageHeight : imagewWidth;
//            int rotate_h = rotation % 180 == 0 ? imagewWidth : imageHeight;
//
//            imageProcess.fillBytes(planes, yuvBytes);
//            int yRowStride = planes[0].getRowStride();
//            final int uvRowStride = planes[1].getRowStride();
//            final int uvPixelStride = planes[1].getPixelStride();
//
//            int[] rgbBytes = new int[imageHeight * imagewWidth];
//            imageProcess.YUV420ToARGB8888(
//                    yuvBytes[0],
//                    yuvBytes[1],
//                    yuvBytes[2],
//                    imagewWidth,
//                    imageHeight,
//                    yRowStride,
//                    uvRowStride,
//                    uvPixelStride,
//                    rgbBytes);
//
//            // 原图bitmap
//            Bitmap imageBitmap = Bitmap.createBitmap(imagewWidth, imageHeight, Bitmap.Config.ARGB_8888);
//            imageBitmap.setPixels(rgbBytes, 0, imagewWidth, 0, 0, imagewWidth, imageHeight);
////            Log.i("nanodet","image w/h"+imagewWidth+"/"+imagewWidth+ " preview w/h"+previewWidth+"/"+previewHeight);
//
//            // 图片适应屏幕fill_start格式的bitmap
//            double scale = Math.max(
////                    previewHeight / (double) (rotation % 180 == 0 ? imagewWidth : imageHeight),
//                    previewHeight / (double)rotate_h,
////                    previewWidth / (double) (rotation % 180 == 0 ? imageHeight : imagewWidth)
//                    previewWidth / (double)rotate_w
//            );
//            Matrix fullScreenTransform = imageProcess.getTransformationMatrix(
//                    imagewWidth, imageHeight,
//                    (int) (scale * imageHeight), (int) (scale * imagewWidth),
//                    rotation % 180 == 0 ? 90 : 0, false
//            );
//
//            // 适应preview的全尺寸bitmap
//            Bitmap fullImageBitmap = Bitmap.createBitmap(imageBitmap, 0, 0, imagewWidth, imageHeight, fullScreenTransform, false);
//            // 裁剪出跟preview在屏幕上一样大小的bitmap
//            Bitmap cropImageBitmap = Bitmap.createBitmap(fullImageBitmap, 0, 0, previewWidth, previewHeight);
//            Bitmap cropImageBitmap = Bitmap.createBitmap(fullImageBitmap, 0, 0, 300, 300);

            // 模型输入的bitmap
//            Matrix previewToModelTransform =
//                    imageProcess.getTransformationMatrix(
//                            cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
////                            yolov5TFLiteDetector.getInputSize().getWidth(),
//                            300,
////                            yolov5TFLiteDetector.getInputSize().getHeight(),
//                            300,
//                            0, false);
//            Bitmap modelInputBitmap = Bitmap.createBitmap(cropImageBitmap, 0, 0,
//                    cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
//                    previewToModelTransform, false);

//            Matrix modelToPreviewTransform = new Matrix();
//            previewToModelTransform.invert(modelToPreviewTransform);
//            int crop_w = (int) Math.floor(rotate_w * previewWidth / (rotate_w * scale));
//            int crop_h = (int) Math.floor(rotate_h * previewHeight / (rotate_h * scale));
//            Log.i("nanode", "crop wh "+crop_w+"/"+crop_h+" rotate wh "+previewWidth+"/"+previewHeight);
//            NanodetplusNcnnDetector.BoxInfo[] recognitions = nanodetplusNcnnDetector.detect(
//                    imageBitmap,
//                    nanodetplusNcnnDetector.NUM_CLASSES,
//                    rotation % 180 == 0 ? 90 : 0,
//                    crop_w,
//                    crop_h,
//                    previewWidth,
//                    previewHeight
//            );
//            NanodetplusNcnnDetector.BoxInfo[] recognitions = nanodetplusNcnnDetector.detect(cropImageBitmap, nanodetplusNcnnDetector.NUM_CLASSES);
//            Bitmap yuvBytes2 = imageProcess.yuv420ToNv21(image);
            byte[] yuvBytes2 = imageProcess.yuv420ToNv21(image);
//            Bitmap nv21Bitmap = imageProcess.nv21ToBitmap(yuvBytes2, imagewWidth, imageHeight);
//            Bitmap imageProxyBitmap = imageProcess.imageProxy2Bitmap(image);
//            Bitmap yuv_bitmap = BitmapFactory.decodeByteArray(yuvBytes2, 0, yuvBytes2.length);
//            Bitmap cropImageBitmap = Bitmap.createBitmap(yuv_bitmap, 0, 0, 400, 400);
            NanodetplusNcnnDetector.BoxInfo[] recognitions = nanodetplusNcnnDetector.detect_yuv(yuvBytes2, imagewWidth, imageHeight, nanodetplusNcnnDetector.NUM_CLASSES);
//            NanodetplusNcnnDetector.BoxInfo[] recognitions = nanodetplusNcnnDetector.detect(cropImageBitmap, true, nanodetplusNcnnDetector.NUM_CLASSES);
//            ArrayList<Recognition> recognitions = yolov5TFLiteDetector.detect(imageBitmap);

            Bitmap emptyCropSizeBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
            Canvas cropCanvas = new Canvas(emptyCropSizeBitmap);
//            Paint white = new Paint();
//            white.setColor(Color.WHITE);
//            white.setStyle(Paint.Style.FILL);
//            cropCanvas.drawRect(new RectF(0,0,previewWidth, previewHeight), white);
            // 边框画笔
            Paint boxPaint = new Paint();
            boxPaint.setStrokeWidth(5);
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setColor(Color.RED);
            // 字体画笔
            Paint textPain = new Paint();
            textPain.setTextSize(50);
            textPain.setColor(Color.RED);
            textPain.setStyle(Paint.Style.FILL);

            Log.i("nanodet:", "recognitions size: "+recognitions.length);
            for (NanodetplusNcnnDetector.BoxInfo res : recognitions) {
//                Log.i("ncnn:", res.toString());
                RectF location = new RectF();
                location.left = res.x1;
                location.top = res.y1;
                location.right = res.x2;
                location.bottom = res.y2;
                float confidence = res.score;
//                modelToPreviewTransform.mapRect(location);
                cropCanvas.drawRect(location, boxPaint);
                String label = nanodetplusNcnnDetector.getLabel(res.label);
                cropCanvas.drawText(label + ":" + String.format("%.2f", confidence), location.left, location.top, textPain);
            }
            long end = System.currentTimeMillis();
            long costTime = (end - start);
            image.close();
            emitter.onNext(new Result(costTime, emptyCropSizeBitmap));
//            emitter.onNext(new Result(costTime, imageBitmap));

        }).subscribeOn(Schedulers.io()) // 这里定义被观察者,也就是上面代码的线程, 如果没定义就是主线程同步, 非异步
                // 这里就是回到主线程, 观察者接受到emitter发送的数据进行处理
                .observeOn(AndroidSchedulers.mainThread())
//                 这里就是回到主线程处理子线程的回调数据.
                .subscribe((Result result) -> {
                    boxLabelCanvas.setImageBitmap(result.bitmap);
                    frameSizeTextView.setText(previewHeight + "x" + previewWidth);
                    inferenceTimeTextView.setText(Long.toString(result.costTime) + "ms");
                });

    }
}
