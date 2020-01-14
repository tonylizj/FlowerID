'use strict';
import React, { Component } from 'react';
import { AppRegistry, StyleSheet, Text, TouchableOpacity, View, Image, ImageBackground } from 'react-native';
import { RNCamera } from 'react-native-camera';
import styles from './styles';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO, fetch } from '@tensorflow/tfjs-react-native';
import base64 from 'react-native-base64';
import ImageEditor from '@react-native-community/image-editor';
import ImgToBase64 from 'react-native-image-base64';
var jpeg = require('jpeg-js');
var RNFS = require('react-native-fs');
const classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'];
/*
const modelJson = RNFS.readFileAssets('model/model.json','base64');
const modelWeights = RNFS.readFileAssets('model/model.bin','base64');
*/
const modelJson = require('../android/app/src/main/assets/model/model.json');
const modelWeights = require('../android/app/src/main/assets/model/model.bin');


export default class CameraPage extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isTfReady: false,
      modelReady: false,
      prediction: [],
      image: null,
      imageUri: null,
      captured: false,
      model: null,
      predicted: false,
    };
  }

  async componentDidMount() {
    await tf.ready()
    this.setState({
      isTfReady: true
    })

    const model = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
    /*
    const model = await mobilenet.load();
    */
    console.log("ready");
    this.setState ({ modelReady: true, model: model })
  }

  imageToTensor(rawImageData: ArrayBuffer): tf.Tensor3D {
    const TO_UINT8ARRAY = true;
    const jpegData = Buffer.from(rawImageData.base64 ,'base64');
    const { width, height, data } = jpeg.decode(jpegData, TO_UINT8ARRAY);
    // Drop the alpha channel info for mobilenet
    const buffer = new Uint8Array(200 * 200 * 3);
    let offset = 0; // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];

      offset += 4;
    }
    return tf.tensor4d(buffer, [1, 200, 200, 3]);
  }

  inference = async() => {
    const { image, model } = this.state;
    /*
    const imageAssetPath = Image.resolveAssetSource(image)
    const response = await fetch(image, {}, { isBinary: true });
    const rawImageData = await response.arrayBuffer();
    const imageTensor = this.imageToTensor(rawImageData)
    */
    var imageTensor = this.imageToTensor(image);
    /*
    const imageTensor = this.imgToBlob()
    */
    const pred = await model.predict(imageTensor);
    const probas = pred.arraySync();
    const winner = probas[0].map((x, i) => [x, i]).reduce((r ,a) => (a[0] > r[0] ? a : r))[1];

    console.log(pred);
    console.log(probas);
    console.log(classes[winner]);

    this.setState({
      prediction: classes[winner],
      predicted: true,
    })
  }

  renderPrediction() {
    const { prediction, imageUri, predicted } = this.state;
    return (
      <ImageBackground source={ {uri: imageUri} } style={styles.prediction}>
      {predicted ?
        <View>
          <Text style={styles.resultTextHeader}>Results</Text>
              <View>
                <Text style={styles.resultClass}>{prediction}</Text>
                <TouchableOpacity style={styles.capture} onPress={() => this.setState({ predicted: false, captured: false })}>
                  <View styles={styles.captureBtn}></View>
                </TouchableOpacity>
              </View>
        </View>
      :
      <View style={styles.resultTextHeader}>
        <Text> Predicting </Text>
      </View>
     }
    </ImageBackground>
    );
  }

  takePicture = async() => {
    if (this.camera) {
      const options = { quality: 0.5, base64: true,
                        width: 250};
      const data = await this.camera.takePictureAsync(options);
      console.log(data);


      /*
      var cropUrl;

      const cropData = {
        offset: {x: 0, y: 0},
        size: {width: data.width, height: data.height},
        //displaySize: {width: 200, height: 200}

      }
      ImageEditor.cropImage(data.uri, cropData).then(url => cropUrl = url)
      console.log(cropUrl)
      var newimg;
      ImgToBase64.getBase64String(cropUrl).then(base64String => newimg = base64String);
      console.log(newimg)


      /*
      RNFS.moveFile(data.uri, path);
      const file = data.uri.split("/");
      const path = "file:///sdcard/Android/data/com.flowerid/files/" + file[file.length - 1];
      */
      this.setState({ captured: true, image: data, imageUri: data.uri });
      this.inference();
    }
  };

  render() {
    const { modelReady, isTfReady, captured } = this.state;
      if (captured) {
        return this.renderPrediction();
      }
      return (
          <View style={styles.container}>
              <RNCamera
                ref={ref => {
                  this.camera = ref;
                }}
                style={styles.preview}
                type={RNCamera.Constants.Type.back}
                flashMode={RNCamera.Constants.FlashMode.off}
                captureAudio={false}
                androidCameraPermissionOptions={{
                  title: 'Permission to use camera',
                  message: 'FlowerID need your permission to use your camera',
                  buttonPositive: 'Ok',
                  buttonNegative: 'Cancel',
                }}
              />
              {modelReady ?
              <View style={{ flex: 0, flexDirection: 'row', justifyContent: 'center' }}>
                <TouchableOpacity onPress={this.takePicture.bind(this)} style={styles.capture}>
                  <View style={styles.captureBtn}></View>
                </TouchableOpacity>
              </View>
              :
              <Text> Loading </Text>}
            </View>
      );
  }
}
