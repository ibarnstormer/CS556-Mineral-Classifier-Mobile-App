package com.cs556.mineralclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.icu.text.DecimalFormat
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.cs556.mineralclassifier.databinding.ActivityMainBinding
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.time.DurationUnit
import kotlin.time.measureTime

typealias MineralListener = (mine: List<Float>) -> Unit

class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding

    private var imageCapture: ImageCapture? = null

    lateinit var model: Module

    private lateinit var cameraExecutor: ExecutorService

    private var realTimeMode: Boolean = false

    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions())
        { permissions ->
            // Handle Permission granted/rejected
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && !it.value)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT).show()
            } else {
                startCamera()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        // Set up the listeners for take photo and video capture buttons
        viewBinding.imageCaptureButton.setOnClickListener { takePhoto() }
        viewBinding.switchButton.setOnClickListener { switch() }

        cameraExecutor = Executors.newSingleThreadExecutor()

        model = Module.load(loadFileFromRawResource(R.raw.mineralcnn_dsc_4_21_2025))
    }

    private fun switch() {
        lateinit var text: String
        if (realTimeMode) {
            viewBinding.imageCaptureButton.visibility = View.VISIBLE
            text = "Real Time Mode"
        }
        else {
            viewBinding.imageCaptureButton.visibility = View.INVISIBLE
            text = "Pause"
        }
        viewBinding.switchButton.text = text
        realTimeMode = !realTimeMode
    }

    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }

                override fun
                        onCaptureSuccess(image: ImageProxy){
                    lateinit var result: Array<String>
                    val elapsed = measureTime { val msg = "Photo capture succeeded: ${image.imageInfo}"
                        Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                        Log.d(TAG, msg)

                        val bitmap = Bitmap.createScaledBitmap(image.toBitmap(), 224, 224, false)

                        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                            bitmap,
                            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                            TensorImageUtils.TORCHVISION_NORM_STD_RGB)
                        val inputEValue: EValue = EValue.from(inputTensor)
                        val output: Array<EValue> = model.forward(inputEValue)
                        val scores: FloatArray = output[0].toTensor().dataAsFloatArray

                        result = mineralClass(scores.asList())
                        Log.d(TAG, "scores = ${scores.contentToString()}")
                        Log.d(TAG, "mineral = ${result[0]}")
                    }.toString(DurationUnit.MILLISECONDS,2)

                    val latency = "\nLatency: $elapsed"
                    val text = "Class: ${result[0]} \nConfident: ${result[1]}%" + latency
                    viewBinding.logView.text = text

                    image.close()
                }
            }
        )
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.surfaceProvider = viewBinding.viewFinder.surfaceProvider
                }

            imageCapture = ImageCapture.Builder().build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, MineralAnalyzer(this.model) { list ->
                        if(realTimeMode) {
                            val result = mineralClass(list.dropLast(1))
                            val mineral = "Class: ${result[0]}"
                            val confident = "Confident = ${result[1]}%"
                            val latency = "Latency = ${list.last()} ms"
                            val text = mineral + "\n" + confident + "\n" + latency
                            viewBinding.logView.text = text

                            Log.d(TAG, mineral)
                            Log.d(TAG, confident)
                            Log.d(TAG, latency)
                        }
                    })
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                    // Unbind use cases before rebinding
                    cameraProvider.unbindAll()

                    // Bind use cases to camera
                    cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageCapture, imageAnalyzer
                    )

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "Mineral"
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
            }.toTypedArray()
    }

    private class MineralAnalyzer(private val model: Module, private val listener: MineralListener) : ImageAnalysis.Analyzer {


        override fun analyze(image: ImageProxy) {

            lateinit var scores: FloatArray
            val elapsed = measureTime {
                val bitmap = Bitmap.createScaledBitmap(image.toBitmap(), 224, 224, false)

                val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                    bitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB)
                val inputEValue: EValue = EValue.from(inputTensor)
                val output: Array<EValue> = model.forward(inputEValue)
                scores = output[0].toTensor().dataAsFloatArray
            }.toString(DurationUnit.MILLISECONDS,2).substringBefore("ms").toFloat()

            val returnList = scores.toList() + elapsed

            listener(returnList)
            
            image.close()
        }
    }

    private fun loadFileFromRawResource(resourceId: Int): String {
        this.resources.openRawResource(resourceId).use { inputStream ->
            val file = File(
                this.filesDir,
                this.resources.getResourceEntryName(resourceId) + ".pte"
            )
            FileOutputStream(file).use { outputStream ->
                inputStream.copyTo(outputStream)
            }
            return file.absolutePath
        }
    }

    private fun mineralClass(scores: List<Float>): Array<String> {
        val minerals = arrayOf("Agate","Amethyst","Beryl","Copper","Diopside","Gold",
            "Quartz","Silver","Spinel","Topaz")
        val mineClass = argMax(scores)
        val confident = softMax(scores)[mineClass]
        val df = DecimalFormat("##.##%")
        return arrayOf(minerals[mineClass], df.format(confident))
    }

    private fun <T : Comparable<T>> argMax(list: List<T>): Int {
        return list.indexOf(list.maxOrNull())
    }

    private fun softMax(list: List<Float>): List<Double> {
        val means = list.average()
        val exp = list.map { exp(it - means) }
        val sum = exp.sum()

        return exp.map { it / sum }
    }

}