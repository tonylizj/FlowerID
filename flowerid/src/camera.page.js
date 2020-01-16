'use strict';
import React, { Component } from 'react';
import { AppRegistry, StyleSheet, Text, TouchableOpacity, View, Image, ImageBackground } from 'react-native';
import { RNCamera } from 'react-native-camera';
import styles from './styles';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO, fetch } from '@tensorflow/tfjs-react-native';
import base64 from 'react-native-base64';

var jpeg = require('jpeg-js');
var RNFS = require('react-native-fs');

/*
var modelJson, modelWeights;
RNFS.readFileAssets('model/model.json').then((file) => modelJson = JSON.parse(file));
RNFS.readFileAssets('model/model.bin').then((file) => modelWeights = file);
*/
const modelJson = require('./model/model.json');
const modelWeights = require('./model/model.bin');
const classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'];


export default class CameraPage extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isTfReady: false,
      modelReady: false,
      prediction: [],
      image: null,
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
    console.log(modelJson)
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
    const winner = classes[pred.argMax().dataSync()[0]];
    console.log(winner);
    this.setState({
      prediction: winner,
      predicted: true,
    })
  }

  renderPrediction() {
    const { prediction, image, predicted } = this.state;
    return (
      <ImageBackground source={ {uri: image.uri} } style={styles.prediction}>
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
      const options = { quality: 0.5, base64: true };
      const data = await this.camera.takePictureAsync(options);
      console.log(data.uri);
      const file = data.uri.split("/");
      const path = "file:///sdcard/Android/data/com.flowerid/files/" + file[file.length - 1];
      /*
      RNFS.moveFile(data.uri, path);
      */
      this.setState({ captured: true, image: data });
      await this.inference();
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
