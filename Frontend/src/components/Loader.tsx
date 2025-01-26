// In your Loader.tsx
import React from 'react';
import {StyleSheet, ImageBackground, Text, View} from 'react-native';

const Loader: React.FC = () => {
  return (
    <ImageBackground
      source={require('./newscollage.jpg')} // Adjust path to your image location
      style={styles.container}
      imageStyle={{opacity: 0.6}}>
      <View style={styles.textContainer}>
        <Text style={styles.appName}>NewsNavigator</Text>
      </View>
    </ImageBackground>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  textContainer: {
    position: 'absolute',
    justifyContent: 'center',
    alignItems: 'center',
  },
  appName: {
    fontSize: 36,
    fontWeight: 'bold',
    color: 'black', // Adjust color as needed
    textAlign: 'center',
  },
});

export default Loader;
