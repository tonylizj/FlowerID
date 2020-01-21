import { StyleSheet, Dimensions } from 'react-native';

const { width: winWidth, height: winHeight } = Dimensions.get('window');

export default StyleSheet.create({
    container: {
      flex: 1,
      flexDirection: 'column',
      backgroundColor: 'black',
    },
    preview: {
      flex: 1,
      justifyContent: 'flex-end',
      alignItems: 'center',
    },
    textBox: {
      flex: 0,
      backgroundColor: '#fff',
      borderRadius: 5,
      padding: 15,
      paddingHorizontal: 20,
      alignSelf: 'center',
      margin: 20,
    },
    captureBtn: {
      width: 60,
      height: 60,
      borderWidth: 2,
      borderRadius: 60,
      borderColor: "#FFFFFF",
    },
    resultTextHeader: {
      fontSize: 21,
      marginBottom: 6,
      marginTop: 6,
      textAlign: 'center',
      borderWidth: 5,
      borderRadius: 100,
      borderColor: "#fff",
      backgroundColor: '#fff'
    },
    prediction: {
      width: '100%',
      height: '100%',
    },
    resultClass: {
      fontSize: 16,
      fontWeight: 'bold'
    },
    resultProb: {
      fontSize: 16,
      marginLeft: 5
    },
    results: {
      flex: 1,
      flexDirection: 'column',
      padding: 15,
      paddingHorizontal: 20,
      alignItems: 'center',
      alignSelf: 'center',
      margin: 20,
      justifyContent: 'flex-end',
    },
  });
