'use strict';

import React, { Component } from 'react';
import { AppRegistry, StyleSheet, Text, TouchableOpacity, View, Image, ImageBackground, BackHandler } from 'react-native';
import { RNCamera } from 'react-native-camera';
import styles from './styles';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as IM from 'expo-image-manipulator';

var jpeg = require('jpeg-js');

const modelWeights = require('@model/model.bin');
const modelJson = require('@model/model.json');

const classes = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip'];

export default class CameraPage extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isTfReady: false,
      modelReady: false,
      prediction: [],
      image: null,
      base64: null,
      captured: false,
      model: null,
      predicted: false,
    };
  }

  async componentDidMount() {
    await tf.ready()
    const model = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
    this.setState ({ modelReady: true, model: model, isTfReady: true })
  }


  imageToTensor(rawImageString): tf.Tensor4D {
    const TO_UINT8ARRAY = true;
    const jpegData = Buffer.from(rawImageString ,'base64');
    const { width, height, data } = jpeg.decode(jpegData, TO_UINT8ARRAY);
    // Drop the alpha channel info for mobilenet
    const buffer = new Uint8Array(width * height * 3);
    let offset = 0; // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];

      offset += 4;
    }
    return tf.tensor4d(buffer, [1, width, height, 3]);
  }

  getMax(array) {
    return [].map.call(array, (x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
  }

  inference = async() => {
    const { model, base64 } = this.state;
    var imageTensor = this.imageToTensor(base64);
    const pred = await model.predict(imageTensor);
    const winner = pred.dataSync();
    console.log(winner)
    console.log(this.getMax(winner));
    console.log(pred.print());
    this.setState({
      prediction: classes[this.getMax(winner)],
      predicted: true,
    })
  }

  takePicture = async() => {
      const options = { quality: 0.5, base64: true };
      const data = await this.camera.takePictureAsync(options);
      const { uri, width, height, base64 } = await IM.manipulateAsync(data.uri, [{resize: { width: 200, height: 200}}], { base64: true });
      this.setState({ captured: true, image: data, base64: base64 });
      await this.inference();
  }

  renderPrediction() {
    const { prediction, image, predicted } = this.state;
    return (
      <ImageBackground source={ {uri: image.uri} } style={styles.prediction}>
      {predicted ?
        <View style={styles.results}>
          <Text style={styles.resultTextHeader}> Results</Text>
          <Text style={styles.resultTextHeader}>{prediction}</Text>
          <TouchableOpacity onPress={() => this.setState({ predicted: false, captured: false} )} style={styles.textBox}>
            <View style={styles.captureBtn}>
              <Text> Go Back </Text>
            </View>
          </TouchableOpacity>
        </View>
      :
      <View style={styles.textBox}>
        <Text> Predicting </Text>
      </View>
     }
    </ImageBackground>
    );
  }

  render() {
    const { modelReady, isTfReady, captured } = this.state;
    if (captured) {
      return this.renderPrediction();
    }
    else {
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

          <View style={styles.textBox}>
          {modelReady ?
            <TouchableOpacity onPress={this.takePicture.bind(this)}>
              <View style={styles.captureBtn}></View>
            </TouchableOpacity>
          :
              <Text> Loading Model </Text>
          }
          </View>
        </View>
    );
  }
  }
}
