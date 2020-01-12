'use strict';
import React, { Component } from 'react';
import { AppRegistry, StyleSheet, Text, TouchableOpacity, View, Image } from 'react-native';
import { RNCamera } from 'react-native-camera';
import styles from './styles';
/*
import Tflite from 'tflite-react-native';
*/
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO, fetch, decodeJpeg } from '@tensorflow/tfjs-react-native';
const modelJson = require('./model/model.json');
const modelWeights = require('./model/model.bin');
import * as mobilenet from '@tensorflow-models/mobilenet';

var RNFS = require('react-native-fs');


export default class CameraPage extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isTfReady: false,
      modelReady: false,
      predictions: null,
      imagePath: null,
      captured: false,
      model: null,
    };
  }

  async componentDidMount() {
    await tf.ready()
    this.setState({
      isTfReady: true
    })
    /*
    const model = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
    */
    const model = await mobilenet.load();
    this.setState ({ modelReady: true, model: model })
  }

  inference = async() => {
    const { image, model } = this.state;
    const imageAssetPath = Image.resolveAssetSource(image);
    const response = await fetch(imageAssetPath.uri, {}, { isBinary: true });
    const rawImageData = await response.arrayBuffer();
    const imageTensor = decodeJpeg(rawImageData);
    const pred = await this.model.predict(imageTensor);
    this.setState({
      predictions: pred,
    })
  }

  takePicture = async() => {
    if (this.camera) {
      const options = { quality: 0.5, base64: true };
      const data = await this.camera.takePictureAsync(options);
      const filepath = data.uri.split("/");
      const name = filepath[filepath.length - 1];
      RNFS.moveFile(data.uri, "file:///sdcard/Android/data/com.flowerid/files/" + name);
      this.setState({ captured: true, image: data });
    }
  };

  render() {
    const { modelReady, isTfReady, captured } = this.state;
    return (
      <View>
      {modelReady ?
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
            <View style={{ flex: 0, flexDirection: 'row', justifyContent: 'center' }}>
            <TouchableOpacity onPress={this.takePicture.bind(this)} style={styles.capture}>
              <View style={styles.captureBtn}></View>
            </TouchableOpacity>
          </View>
        </View>
    :
    <Text> Loading </Text> }
    </View>
    );
  }


}
