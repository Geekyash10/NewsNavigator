import axios from 'axios';
import React, {useState} from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native';
import LottieView from 'lottie-react-native';

const SignupScreen = ({navigation}) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSignup = async () => {
    try {
      const response = await axios.post('http://10.38.154.157:5001/signup', {
        email,
        password,
      });

      if (response.status === 201) {
        Alert.alert('Signup Successful', 'You can now log in.');
        navigation.navigate('Login');
      } else {
        Alert.alert(
          'Signup Failed',
          response.data.message || 'An error occurred during signup.',
        );
      }
    } catch (error) {
      if (error.response) {
        Alert.alert(
          'Signup Error',
          error.response.data.message || 'An error occurred on the server.',
        );
      } else if (error.request) {
        Alert.alert(
          'Signup Error',
          'Server did not respond. Check your network connection.',
        );
      } else {
        Alert.alert('Signup Error', error.message);
      }
    }
  };

  return (
    <View style={styles.container}>
      <LottieView
        source={require('./loginscreen.json')}
        autoPlay
        loop
        style={styles.lottie}
      />
      <Text style={styles.title}>Sign Up</Text>
      <TextInput
        style={styles.input}
        placeholderTextColor="#888"
        placeholder="Email"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        placeholderTextColor="#888"
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />
      <TouchableOpacity style={styles.button} onPress={handleSignup}>
        <Text style={styles.buttonText}>Sign Up</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    padding: 16,
    backgroundColor: '#f3f4f6',
  },
  lottie: {
    width: 350,
    height: 400,
    alignSelf: 'center',
    marginBottom: 10,
  },
  title: {
    fontSize: 26,
    fontWeight: '600',
    textAlign: 'center',
    color: '#333',
    marginBottom: 20,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    backgroundColor: '#fff',
    color: '#000000',
  },
  button: {
    backgroundColor: '#4a90e2',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 10,
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
});

export default SignupScreen;
