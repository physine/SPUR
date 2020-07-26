/*
 * Copyright 2017 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ar.core.examples.java.helloar;

import android.content.DialogInterface;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.Image;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Trace;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Script;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.util.Base64;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageButton;
import android.widget.Toast;
import com.google.ar.core.Anchor;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Coordinates2d;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Point;
import com.google.ar.core.Point.OrientationMode;
import com.google.ar.core.PointCloud;
import com.google.ar.core.Session;
import com.google.ar.core.Trackable;
import com.google.ar.core.TrackingState;
import com.google.ar.core.examples.java.common.helpers.CameraPermissionHelper;
import com.google.ar.core.examples.java.common.helpers.DepthSettings;
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper;
import com.google.ar.core.examples.java.common.helpers.FullScreenHelper;
import com.google.ar.core.examples.java.common.helpers.SnackbarHelper;
import com.google.ar.core.examples.java.common.helpers.TapHelper;
import com.google.ar.core.examples.java.common.helpers.TrackingStateHelper;
import com.google.ar.core.examples.java.common.rendering.BackgroundRenderer;
import com.google.ar.core.examples.java.common.rendering.ObjectRenderer;
import com.google.ar.core.examples.java.common.rendering.ObjectRenderer.BlendMode;
import com.google.ar.core.examples.java.common.rendering.PlaneRenderer;
import com.google.ar.core.examples.java.common.rendering.PointCloudRenderer;
import com.google.ar.core.examples.java.common.rendering.Texture;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import org.tensorflow.lite.Interpreter;

import android.renderscript.ScriptC;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.Buffer;

/**
 * This is a simple example that shows how to create an augmented reality (AR) application using the
 * ARCore API. The application will display any detected planes and will allow the user to tap on a
 * plane to place a 3d model of the Android robot.
 */
public class HelloArActivity extends AppCompatActivity implements Classifier, GLSurfaceView.Renderer {
  private static final String TAG = HelloArActivity.class.getSimpleName();

  // Rendering. The Renderers are created here, and initialized when the GL surface is created.
  private GLSurfaceView surfaceView;

  private boolean installRequested;

  private Session session;
  private final SnackbarHelper messageSnackbarHelper = new SnackbarHelper();
  private DisplayRotationHelper displayRotationHelper;
  private final TrackingStateHelper trackingStateHelper = new TrackingStateHelper(this);
  private TapHelper tapHelper;

  private final BackgroundRenderer backgroundRenderer = new BackgroundRenderer();
  private final ObjectRenderer virtualObject = new ObjectRenderer();
  private final PlaneRenderer planeRenderer = new PlaneRenderer();
  private final PointCloudRenderer pointCloudRenderer = new PointCloudRenderer();
  private final Texture depthTexture = new Texture();
  private boolean calculateUVTransform = true;

  private final DepthSettings depthSettings = new DepthSettings();
  private boolean[] settingsMenuDialogCheckboxes;

  // Temporary matrix allocated here to reduce number of allocations for each frame.
  private final float[] anchorMatrix = new float[16];
  private static final float[] DEFAULT_COLOR = new float[] {0f, 0f, 0f, 0f};

  private static final String SEARCHING_PLANE_MESSAGE = "Searching for surfaces...";

  // private static final Logger LOGGER = new Logger();

  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;
  // Float model
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize = 300;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;
  private ByteBuffer imgData;
  private Interpreter tfLite;


  private MappedByteBuffer loadModelFile() throws IOException {
    AssetFileDescriptor fileDescriptor = getAssets().openFd("detect.tflite");
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        return null;
    }

    @Override
    public void enableStatLogging(boolean debug) {

    }

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {
    }

    @Override
    public void setNumThreads(int num_threads) {
        if (tfLite != null) tfLite.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        if (tfLite != null) tfLite.setUseNNAPI(isChecked);
    }


    // Anchors created from taps used for object placing with a given color.
  private static class ColoredAnchor {
    public final Anchor anchor;
    public final float[] color;

    public ColoredAnchor(Anchor a, float[] color4f) {
      this.anchor = a;
      this.color = color4f;
    }
  }

  private final ArrayList<ColoredAnchor> anchors = new ArrayList<>();

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    setContentView(R.layout.activity_main);
    surfaceView = findViewById(R.id.surfaceview);
    displayRotationHelper = new DisplayRotationHelper(/*context=*/ this);

    // Set up tap listener.
    tapHelper = new TapHelper(/*context=*/ this);
    surfaceView.setOnTouchListener(tapHelper);

    // Set up renderer.
    surfaceView.setPreserveEGLContextOnPause(true);
    surfaceView.setEGLContextClientVersion(2);
    surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0); // Alpha used for plane blending.
    surfaceView.setRenderer(this);
    surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
    surfaceView.setWillNotDraw(false);

    installRequested = false;
    calculateUVTransform = true;

    depthSettings.onCreate(this);
    ImageButton settingsButton = findViewById(R.id.settings_button);
    settingsButton.setOnClickListener(this::launchSettingsMenuDialog);


    // tfLite stuff
    AssetManager assetManager = getAssets();

    InputStream labelsInput = null;
    try {
      labelsInput = assetManager.open("labelmap.txt");
    } catch (IOException e) {
      e.printStackTrace();
    }

    BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
    String line = null;
    while (true) {
      try {
        if (!((line = br.readLine()) != null)) break;
      } catch (IOException e) {
        e.printStackTrace();
      }
      //LOGGER.w(line);
      labels.add(line);
    }
    try {
      br.close();
    } catch (IOException e) {
      e.printStackTrace();
    }


    try{
      // AssetManager assets, String modelFilename
      tfLite = new Interpreter(loadModelFile());
      Log.i("***** Success ", " Loaded Model ****");
    }catch(Exception e){
      e.printStackTrace();
    }

    boolean isQuantized = true;
    isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * numBytesPerChannel);
    imgData.order(ByteOrder.nativeOrder());
    intValues = new int[inputSize * inputSize];

    tfLite.setNumThreads(NUM_THREADS);
    outputLocations = new float[1][NUM_DETECTIONS][4];
    outputClasses = new float[1][NUM_DETECTIONS];
    outputScores = new float[1][NUM_DETECTIONS];
    numDetections = new float[1];

  }

  @Override
  protected void onResume() {
    super.onResume();

    if (session == null) {
      Exception exception = null;
      String message = null;
      try {
        switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
          case INSTALL_REQUESTED:
            installRequested = true;
            return;
          case INSTALLED:
            break;
        }

        // ARCore requires camera permissions to operate. If we did not yet obtain runtime
        // permission on Android M and above, now is a good time to ask the user for it.
        if (!CameraPermissionHelper.hasCameraPermission(this)) {
          CameraPermissionHelper.requestCameraPermission(this);
          return;
        }

        // Create the session.
        session = new Session(/* context= */ this);
        Config config = session.getConfig();
        if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
          config.setDepthMode(Config.DepthMode.AUTOMATIC);
        } else {
          config.setDepthMode(Config.DepthMode.DISABLED);
        }
        session.configure(config);
      } catch (UnavailableArcoreNotInstalledException
          | UnavailableUserDeclinedInstallationException e) {
        message = "Please install ARCore";
        exception = e;
      } catch (UnavailableApkTooOldException e) {
        message = "Please update ARCore";
        exception = e;
      } catch (UnavailableSdkTooOldException e) {
        message = "Please update this app";
        exception = e;
      } catch (UnavailableDeviceNotCompatibleException e) {
        message = "This device does not support AR";
        exception = e;
      } catch (Exception e) {
        message = "Failed to create AR session";
        exception = e;
      }

      if (message != null) {
        messageSnackbarHelper.showError(this, message);
        Log.e(TAG, "Exception creating session", exception);
        return;
      }
    }

    // Note that order matters - see the note in onPause(), the reverse applies here.
    try {
      session.resume();
    } catch (CameraNotAvailableException e) {
      messageSnackbarHelper.showError(this, "Camera not available. Try restarting the app.");
      session = null;
      return;
    }

    surfaceView.onResume();
    displayRotationHelper.onResume();
  }

  @Override
  public void onPause() {
    super.onPause();
    if (session != null) {
      // Note that the order matters - GLSurfaceView is paused first so that it does not try
      // to query the session. If Session is paused before GLSurfaceView, GLSurfaceView may
      // still call session.update() and get a SessionPausedException.
      displayRotationHelper.onPause();
      surfaceView.onPause();
      session.pause();
    }
  }

  @Override
  public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
    super.onRequestPermissionsResult(requestCode, permissions, results);
    if (!CameraPermissionHelper.hasCameraPermission(this)) {
      Toast.makeText(this, "Camera permission is needed to run this application", Toast.LENGTH_LONG)
          .show();
      if (!CameraPermissionHelper.shouldShowRequestPermissionRationale(this)) {
        // Permission denied with checking "Do not ask again".
        CameraPermissionHelper.launchPermissionSettings(this);
      }
      finish();
    }
  }

  @Override
  public void onWindowFocusChanged(boolean hasFocus) {
    super.onWindowFocusChanged(hasFocus);
    FullScreenHelper.setFullScreenOnWindowFocusChanged(this, hasFocus);
  }

  @Override
  public void onSurfaceCreated(GL10 gl, EGLConfig config) {
    GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // Prepare the rendering objects. This involves reading shaders, so may throw an IOException.
    try {
      // Create the texture and pass it to ARCore session to be filled during update().
      depthTexture.createOnGlThread();
      backgroundRenderer.createOnGlThread(/*context=*/ this, depthTexture.getTextureId());
      planeRenderer.createOnGlThread(/*context=*/ this, "models/trigrid.png");
      pointCloudRenderer.createOnGlThread(/*context=*/ this);

      virtualObject.createOnGlThread(/*context=*/ this, "models/andy.obj", "models/andy.png");
      virtualObject.setBlendMode(BlendMode.AlphaBlending);
      virtualObject.setDepthTexture(
          depthTexture.getTextureId(), depthTexture.getWidth(), depthTexture.getHeight());
      virtualObject.setMaterialProperties(0.0f, 2.0f, 0.5f, 6.0f);

    } catch (IOException e) {
      Log.e(TAG, "Failed to read an asset file", e);
    }
  }

  @Override
  public void onSurfaceChanged(GL10 gl, int width, int height) {
    displayRotationHelper.onSurfaceChanged(width, height);
    GLES20.glViewport(0, 0, width, height);
  }

  @Override
  public void onDrawFrame(GL10 gl) {
    // Clear screen to notify driver it should not load any pixels from previous frame.
    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

    if (session == null) {
      return;
    }
    // Notify ARCore session that the view size changed so that the perspective matrix and
    // the video background can be properly adjusted.
    displayRotationHelper.updateSessionIfNeeded(session);

    try {
      session.setCameraTextureName(backgroundRenderer.getTextureId());

      // Obtain the current frame from ARSession. When the configuration is set to
      // UpdateMode.BLOCKING (it is by default), this will throttle the rendering to the
      // camera framerate.
      Frame frame = session.update();
      Camera camera = frame.getCamera();

      if (frame.hasDisplayGeometryChanged() || calculateUVTransform) {
        // The UV Transform represents the transformation between screenspace in normalized units
        // and screenspace in units of pixels.  Having the size of each pixel is necessary in the
        // virtual object shader, to perform kernel-based blur effects.
        calculateUVTransform = false;
        float[] transform = getTextureTransformMatrix(frame);
        virtualObject.setUvTransformMatrix(transform);
      }

      if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
        depthTexture.updateWithDepthImageOnGlThread(frame);
      }

      // Handle one tap per frame.
      handleTap(frame, camera);

      // If frame is ready, render camera preview image to the GL surface.
      backgroundRenderer.draw(frame, depthSettings.depthColorVisualizationEnabled());

      // Keep the screen unlocked while tracking, but allow it to lock when tracking stops.
      trackingStateHelper.updateKeepScreenOnFlag(camera.getTrackingState());

      // If not tracking, don't draw 3D objects, show tracking failure reason instead.
      if (camera.getTrackingState() == TrackingState.PAUSED) {
        messageSnackbarHelper.showMessage( // "moving to fast, slow down"
            this, TrackingStateHelper.getTrackingFailureReasonString(camera));
        return;
      }

      // Get projection matrix.
      float[] projmtx = new float[16];
      camera.getProjectionMatrix(projmtx, 0, 0.1f, 100.0f);

      // Get camera matrix and draw.
      float[] viewmtx = new float[16];
      camera.getViewMatrix(viewmtx, 0);

      // Compute lighting from average intensity of the image.
      // The first three components are color scaling factors.
      // The last one is the average pixel intensity in gamma space.
      final float[] colorCorrectionRgba = new float[4];
      frame.getLightEstimate().getColorCorrection(colorCorrectionRgba, 0);

      // Visualize tracked points.
      // Use try-with-resources to automatically release the point cloud.
      try (PointCloud pointCloud = frame.acquirePointCloud()) {
        pointCloudRenderer.update(pointCloud);
        pointCloudRenderer.draw(viewmtx, projmtx);
      }

      // No tracking error at this point. If we detected any plane, then hide the
      // message UI, otherwise show searchingPlane message.
      if (hasTrackingPlane()) {
        // messageSnackbarHelper.hide(this);
      } else {
        messageSnackbarHelper.showMessage(this, SEARCHING_PLANE_MESSAGE);
      }

      // Visualize planes.
      planeRenderer.drawPlanes(
          session.getAllTrackables(Plane.class), camera.getDisplayOrientedPose(), projmtx);

      // Visualize anchors created by touch.
      float scaleFactor = 0.25f;
      virtualObject.setUseDepthForOcclusion(this, depthSettings.useDepthForOcclusion());
      for (ColoredAnchor coloredAnchor : anchors) {
        if (coloredAnchor.anchor.getTrackingState() != TrackingState.TRACKING) {
          continue;
        }
        // Get the current pose of an Anchor in world space. The Anchor pose is updated
        // during calls to session.update() as ARCore refines its estimate of the world.
        coloredAnchor.anchor.getPose().toMatrix(anchorMatrix, 0);

        // Update and draw the model and its shadow.
        virtualObject.updateModelMatrix(anchorMatrix, scaleFactor);
        virtualObject.draw(viewmtx, projmtx, colorCorrectionRgba, coloredAnchor.color);
      }

    } catch (Throwable t) {
      // Avoid crashing the application due to unhandled exceptions.
      Log.e(TAG, "Exception on the OpenGL thread", t);
    }
  }

//  private static byte[] YUV_420_888toNV21(Image image) {
//    byte[] nv21;
//    ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
//    ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
//    ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();
//
//    int ySize = yBuffer.remaining();
//    int uSize = uBuffer.remaining();
//    int vSize = vBuffer.remaining();
//
//    nv21 = new byte[ySize + uSize + vSize];
//
//    //U and V are swapped
//    yBuffer.get(nv21, 0, ySize);
//    vBuffer.get(nv21, ySize, vSize);
//    uBuffer.get(nv21, ySize + vSize, uSize);
//
//    return nv21;
//  }

  private Bitmap decodeToBitMap(byte[] data) {
    try {
      YuvImage image = new YuvImage(data, ImageFormat.NV21, 640,
              480, null);
      if (image != null) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        image.compressToJpeg(new Rect(0, 0, 640, 480),
                80, stream);
        Bitmap bmp = BitmapFactory.decodeByteArray(
                stream.toByteArray(), 0, stream.size());
        stream.close();
        return bmp ;
      }
    } catch (Exception ex) {
      Log.e("Sys", "Error:" + ex.getMessage());
    }
    return null;
  }

  public Bitmap YUVtoBitmap(byte[] imageBuffer, int width, int height /*, ScanSession session */) {
    YuvImage image = new YuvImage(imageBuffer, ImageFormat.NV21, width, height, null);
    ByteArrayOutputStream stream = new ByteArrayOutputStream();
    image.compressToJpeg(new Rect(0, 0, width, height), 100, stream);
    byte[] jpegData = stream.toByteArray();
    final Bitmap theBitmap = BitmapFactory.decodeByteArray(jpegData, 0, jpegData.length);
    return theBitmap;
  }

  public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
    int width = bm.getWidth();
    int height = bm.getHeight();
    float scaleWidth = ((float) newWidth) / width;
    float scaleHeight = ((float) newHeight) / height;
    // CREATE A MATRIX FOR THE MANIPULATION
    Matrix matrix = new Matrix();
    // RESIZE THE BIT MAP
    matrix.postScale(scaleWidth, scaleHeight);

    // "RECREATE" THE NEW BITMAP
    Bitmap resizedBitmap = Bitmap.createBitmap(
            bm, 0, 0, width, height, matrix, false);
    bm.recycle();
    return resizedBitmap;
  }

  public void convertBitmapToByteBuffer(Bitmap bitmap){

    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }


//    imgData.rewind();
//    for (int i = 0; i < inputSize; ++i) {
//      for (int j = 0; j < inputSize; ++j) {
//        int pixelValue = intValues[i * inputSize + j];
//        if (isModelQuantized) {
//          // Quantized model
//          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
//          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
//          imgData.put((byte) (pixelValue & 0xFF));
//        } else { // Float model
//          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//        }
//      }
//    }
//    imgData.rewind();
//    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//
//    int pixel = 0;
//    for (int i = 0; i < 300; ++i) {
//      for (int j = 0; j < 300; ++j) {
//        final int val = intValues[pixel++];
//        imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
//        imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
//        imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
////        if (isModelQuantized) {
////          // Quantized model
////          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
////          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
////          imgData.put((byte) (pixelValue & 0xFF));
////        } else { // Float model
////          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
////          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
////          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
////        }
//      }
//    }
  }

  // Handle only one tap per frame, as taps are usually low frequency compared to frame rate.
  private void handleTap(Frame frame, Camera camera) {
    MotionEvent tap = tapHelper.poll();
    if (tap != null && camera.getTrackingState() == TrackingState.TRACKING) {
      try{
        // Step1: get current image
        Image currentImage = frame.acquireCameraImage();
        Log.i("success","**** Success, got current image of frame ****");
        Log.i("pic_format","**** "+currentImage.getFormat()+" ****"); // 35
        Log.i("pic_dimensions","**** width="+currentImage.getWidth()+" height="+currentImage.getHeight()+" ****");
        // convert the img(YUV_420_888) to bitmap width=640 height=480
        // convert dimensions of image and re-size if not 300x300px
        // default on google Pixel A3 is Height=480 Width=640
        // convert image to bitmap before resizing it
        ByteBuffer buffer = currentImage.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.capacity()];
        buffer.get(bytes);
        Bitmap bitmapImage = decodeToBitMap(bytes);
        Log.i("bitmap","**** width="+bitmapImage.getWidth()+" height="+bitmapImage.getHeight()+" ****");

        // resize bitmap
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmapImage, 300, 300, true);
        Log.i("resizedBitmap","**** width="+resizedBitmap.getWidth()+" height="+resizedBitmap.getHeight()+" ****");

        if (bitmapImage == null){
          Log.i("null","**** bm is NULL ****");
        }
        if (resizedBitmap == null){
          Log.i("null","**** resizedBitmap is NULL ****");
        }
        Bitmap input = resizedBitmap;

        convertBitmapToByteBuffer(input);

        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);
        Log.i("I/O","**** inputArray="+inputArray+" ****");
        Log.i("I/O","**** outputMap="+outputMap+" ****");
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        Log.i("Model1", "**** Model run with No Error ****");
        Log.i("Model2", "**** outputMap="+outputMap+" ****");

        int numDetectionsOutput = Math.min(NUM_DETECTIONS, (int) numDetections[0]); // cast from float to integer, use min for safety
        Log.i("Model4", "**** numDetectionsOutput="+numDetectionsOutput+" ****"); // 10

        final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);

        for (int i = 0; i < numDetectionsOutput; ++i) {
          final RectF detection = new RectF(
                          outputLocations[0][i][1] * inputSize,
                          outputLocations[0][i][0] * inputSize,
                         outputLocations[0][i][3] * inputSize,
                       outputLocations[0][i][2] * inputSize);
          int labelOffset = 1;
          recognitions.add(new Recognition("" + i, labels.get((int) outputClasses[0][i] + labelOffset), outputScores[0][i], detection));
        }

        float alpha = 480/300; // scale factor for height
        float beta = 640/300;  // scale factor for width
        float left = recognitions.get(0).getLocation().left * beta; // b
        float right = recognitions.get(0).getLocation().right * beta; //b
        float top = recognitions.get(0).getLocation().top * alpha; //a
        float bottom = recognitions.get(0).getLocation().bottom * alpha; // a

        float centerX = (right-left)/2;
        float centerY = (top-bottom)/2;

        float[] purple = new float[] {166.0f, 133.0f, 244.0f, 255.0f};
        float[] black = new float[] {66.0f, 53.0f, 44.0f, 255.0f};
        float[] pink = new float[] {255.0f, 133.0f, 244.0f, 255.0f};
        float[] green = new float[] {66.0f, 133.0f, 244.0f, 255.0f};
        float[] white = DEFAULT_COLOR;
        Log.i("up-scales co-ordinate locations",
                   "co-ordinates of "+recognitions.get(0).getTitle()+
                        "\nlocation scaled Left Top= ("+ left +", "+ top+")"+
                        "\nlocation scaled Left Bottom= ("+ left +", "+ bottom + ")"+
                        "\nlocation scaled Right Top= ("+ right +", "+ top + ")"+
                        "\nlocation scaled Right Bottom= ("+ right +", "+ bottom +")\n"
        );

        List<HitResult> hitResultsCenter = frame.hitTest(centerX, centerY);
        Log.i("HitResult", "**** hitResultsLeftBottom size= "+hitResultsCenter.size()+" ****");

        List<HitResult> hitResultsLeftBottom = frame.hitTest(left, bottom);
        Log.i("HitResult", "**** hitResultsLeftBottom size= "+hitResultsLeftBottom.size()+" ****");
//
        List<HitResult> hitResultsRightBottom = frame.hitTest(right, bottom);
        Log.i("HitResult", "**** HitResult size= "+hitResultsRightBottom.size()+" ****");
//
        List<HitResult> hitResultsLeftTop = frame.hitTest(left, top);
        Log.i("HitResult", "**** HitResult size= "+hitResultsLeftTop.size()+" ****");

        List<HitResult> hitResultsRightTop = frame.hitTest(right, top);
        Log.i("HitResult", "**** HitResult size= "+hitResultsRightTop.size()+" ****");

        if(hitResultsLeftBottom.size() > 0 || hitResultsRightBottom.size() > 0
                || hitResultsLeftTop.size() > 0 || hitResultsRightTop.size() > 0) {
          // remove all anchors before placing new ones
          if(anchors.size() > 0){
            anchors.removeAll(anchors);
          }
        }

        int numOfAnchors = 0;
        HitResult hitResult;

        if(hitResultsCenter.size() > 0){
          // only anchor to the closest plane if more than one was returned
          hitResult = hitResultsCenter.get(0);
          anchors.add(new ColoredAnchor(hitResult.createAnchor(), black));
          numOfAnchors++;
        }

        //HitResult hitResult;
        // left bottom
        if(hitResultsLeftBottom.size() > 0){
          // only anchor to the closest plane if more than one
          hitResult = hitResultsLeftBottom.get(0);
          anchors.add(new ColoredAnchor(hitResult.createAnchor(), green));
          numOfAnchors++;
        }

        // left top
        if(hitResultsRightBottom.size() > 0){
          // only anchor to the closest plane if more than one was returned
          hitResult = hitResultsRightBottom.get(0);
          anchors.add(new ColoredAnchor(hitResult.createAnchor(), pink));
          numOfAnchors++;
        }

        // left bottom unacurate
        if(hitResultsLeftTop.size() > 0){
          // only anchor to the closest plane if more than one was returned
          hitResult = hitResultsLeftTop.get(0);
          anchors.add(new ColoredAnchor(hitResult.createAnchor(), white));
          numOfAnchors++;
        }

        // left top
        if(hitResultsRightTop.size() > 0){
          // only anchor to the closest plane if more than one was returned
          hitResult = hitResultsRightTop.get(0);
          anchors.add(new ColoredAnchor(hitResult.createAnchor(), purple));
          numOfAnchors++;
        }

        String anchorInfo;
        if(numOfAnchors == 1){
          anchorInfo = numOfAnchors + " anchor placed";
        }else{
          anchorInfo = numOfAnchors + " anchors placed";
        }
        messageSnackbarHelper.showMessage(this, "detecting a "+recognitions.get(0).getTitle()  + "\n"  + anchorInfo );

        input = null;
        resizedBitmap = null;
        bitmapImage = null;
        bytes = null;
        buffer = null;
        currentImage.close();
      }catch (Exception e){
        Log.i("Error_msg","**** Failed in onHandleTap ****"+e);
      }

      for (HitResult hit : frame.hitTest(tap)) {
        // Check if any plane was hit, and if it was hit inside the plane polygon
        Trackable trackable = hit.getTrackable();
        // Creates an anchor if a plane or an oriented point was hit.
        if ((trackable instanceof Plane
                && ((Plane) trackable).isPoseInPolygon(hit.getHitPose())
                && (PlaneRenderer.calculateDistanceToPlane(hit.getHitPose(), camera.getPose()) > 0))
            || (trackable instanceof Point
                && ((Point) trackable).getOrientationMode()
                    == OrientationMode.ESTIMATED_SURFACE_NORMAL)) {
          // Hits are sorted by depth. Consider only closest hit on a plane or oriented point.
          // Cap the number of objects created. This avoids overloading both the
          // rendering system and ARCore.
          if (anchors.size() >= 20) {
            anchors.get(0).anchor.detach();
            anchors.remove(0);
          }

          // Assign a color to the object for rendering based on the trackable type
          // this anchor attached to. For AR_TRACKABLE_POINT, it's blue color, and
          // for AR_TRACKABLE_PLANE, it's green color.
          float[] objColor;
          if (trackable instanceof Point) {
            objColor = new float[] {66.0f, 133.0f, 244.0f, 255.0f};
          } else if (trackable instanceof Plane) {
            objColor = new float[] {139.0f, 195.0f, 74.0f, 255.0f};
          } else {
            objColor = DEFAULT_COLOR;
          }

          // Adding an Anchor tells ARCore that it should track this position in
          // space. This anchor is created on the Plane to place the 3D model
          // in the correct position relative both to the world and to the plane.
            // TODO: note that the bellow line was commented out to prevent android robots appearing
            // TODO: every time the screen is taped on a detected surface
          // anchors.add(new ColoredAnchor(hit.createAnchor(), objColor));

          // For devices that support the Depth API, shows a dialog to suggest enabling
          // depth-based occlusion. This dialog needs to be spawned on the UI thread.
          this.runOnUiThread(this::showOcclusionDialogIfNeeded);
          break;
        }
      }
    }
  }



  /**
   * Shows a pop-up dialog on the first call, determining whether the user wants to enable
   * depth-based occlusion. The result of this dialog can be retrieved with useDepthForOcclusion().
   */
  private void showOcclusionDialogIfNeeded() {
    boolean isDepthSupported = session.isDepthModeSupported(Config.DepthMode.AUTOMATIC);
    if (!depthSettings.shouldShowDepthEnableDialog() || !isDepthSupported) {
      return; // Don't need to show dialog.
    }

    // Asks the user whether they want to use depth-based occlusion.
    new AlertDialog.Builder(this)
        .setTitle(R.string.options_title_with_depth)
        .setMessage(R.string.depth_use_explanation)
        .setPositiveButton(
            R.string.button_text_enable_depth,
            (DialogInterface dialog, int which) -> {
              depthSettings.setUseDepthForOcclusion(true);
            })
        .setNegativeButton(
            R.string.button_text_disable_depth,
            (DialogInterface dialog, int which) -> {
              depthSettings.setUseDepthForOcclusion(false);
            })
        .show();
  }

  /** Shows checkboxes to the user to facilitate toggling of depth-based effects. */
  private void launchSettingsMenuDialog(View view) {
    // Retrieves the current settings to show in the checkboxes.
    resetSettingsMenuDialogCheckboxes();

    // Shows the dialog to the user.
    Resources resources = getResources();
    if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
      // With depth support, the user can select visualization options.
      new AlertDialog.Builder(this)
          .setTitle(R.string.options_title_with_depth)
          .setMultiChoiceItems(
              resources.getStringArray(R.array.depth_options_array),
              settingsMenuDialogCheckboxes,
              (DialogInterface dialog, int which, boolean isChecked) ->
                  settingsMenuDialogCheckboxes[which] = isChecked)
          .setPositiveButton(
              R.string.done,
              (DialogInterface dialogInterface, int which) -> applySettingsMenuDialogCheckboxes())
          .setNegativeButton(
              android.R.string.cancel,
              (DialogInterface dialog, int which) -> resetSettingsMenuDialogCheckboxes())
          .show();
    } else {
      // Without depth support, no settings are available.
      new AlertDialog.Builder(this)
          .setTitle(R.string.options_title_without_depth)
          .setPositiveButton(
              R.string.done,
              (DialogInterface dialogInterface, int which) -> applySettingsMenuDialogCheckboxes())
          .show();
    }
  }

  private void applySettingsMenuDialogCheckboxes() {
    depthSettings.setUseDepthForOcclusion(settingsMenuDialogCheckboxes[0]);
    depthSettings.setDepthColorVisualizationEnabled(settingsMenuDialogCheckboxes[1]);
  }

  private void resetSettingsMenuDialogCheckboxes() {
    settingsMenuDialogCheckboxes = new boolean[2];
    settingsMenuDialogCheckboxes[0] = depthSettings.useDepthForOcclusion();
    settingsMenuDialogCheckboxes[1] = depthSettings.depthColorVisualizationEnabled();
  }

  /** Checks if we detected at least one plane. */
  private boolean hasTrackingPlane() {
    for (Plane plane : session.getAllTrackables(Plane.class)) {
      if (plane.getTrackingState() == TrackingState.TRACKING) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns a transformation matrix that when applied to screen space uvs makes them match
   * correctly with the quad texture coords used to render the camera feed. It takes into account
   * device orientation.
   */
  private static float[] getTextureTransformMatrix(Frame frame) {
    float[] frameTransform = new float[6];
    float[] uvTransform = new float[9];
    // XY pairs of coordinates in NDC space that constitute the origin and points along the two
    // principal axes.
    float[] ndcBasis = {0, 0, 1, 0, 0, 1};

    // Temporarily store the transformed points into outputTransform.
    frame.transformCoordinates2d(
        Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
        ndcBasis,
        Coordinates2d.TEXTURE_NORMALIZED,
        frameTransform);

    // Convert the transformed points into an affine transform and transpose it.
    float ndcOriginX = frameTransform[0];
    float ndcOriginY = frameTransform[1];
    uvTransform[0] = frameTransform[2] - ndcOriginX;
    uvTransform[1] = frameTransform[3] - ndcOriginY;
    uvTransform[2] = 0;
    uvTransform[3] = frameTransform[4] - ndcOriginX;
    uvTransform[4] = frameTransform[5] - ndcOriginY;
    uvTransform[5] = 0;
    uvTransform[6] = ndcOriginX;
    uvTransform[7] = ndcOriginY;
    uvTransform[8] = 1;

    return uvTransform;
  }
}
