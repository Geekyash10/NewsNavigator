// import React, {useState, useLayoutEffect,useEffect} from 'react';
// import {
//   View,
//   Text,
//   StyleSheet,
//   TouchableOpacity,
//   ScrollView,
//   Alert,
// } from 'react-native';
// import AsyncStorage from '@react-native-async-storage/async-storage';
// import {Card, Button} from 'react-native-paper';
// import {useNavigation} from '@react-navigation/native';
// import InputForm from '../components/InputForm.tsx';
// import axios from 'axios';

// const HomeScreen: React.FC<{setIsLoggedIn: (state: boolean) => void}> = ({
//   setIsLoggedIn,
// }) => {
//   const [activeTab, setActiveTab] = useState<'description' | 'link'>(
//     'description',
//   );
//   const [inputValue, setInputValue] = useState('');
//   const navigation = useNavigation();

//   useLayoutEffect(() => {
//     navigation.setOptions({
//       headerRight: () => (
//         <TouchableOpacity onPress={handleLogout} style={styles.logoutButton}>
//           <Text style={styles.logoutText}>Logout</Text>
//         </TouchableOpacity>
//       ),
//     });
//   }, [navigation]);

//   useEffect(() => {
//     setInputValue('');
//   }, [activeTab]);

//   const handleClassifyDescription = async () => {
//     if (!inputValue) {
//       Alert.alert('Error', 'Please provide a description.');
//       return;
//     }

//     try {
//       const response = await axios.post(
//         'http://10.38.154.157:3000/api/description',
//         {
//           description: inputValue,
//         },
//       );

//       if (response.data && response.data.content) {
//         Alert.alert('Classification Result', response.data.content);
//       } else {
//         Alert.alert('Error', response.data.error || 'Classification failed.');
//       }
//     } catch (error) {
//       console.error('Error during classification:', error);
//       Alert.alert('Error', 'Failed to classify description. Please try again.');
//     }
//   };

//   const handleClassifyLink = async () => {
//     if (!inputValue) {
//       Alert.alert('Error', 'Please provide a link.');
//       return;
//     }

//     try {
//       const response = await axios.post('http://10.38.154.157:3000/api/scrape', {
//         url: inputValue,
//       });

//       if (response.data && response.data.content) {
//         Alert.alert('Classification Result', response.data.content);
//       } else {
//         Alert.alert('Error', response.data.error || 'Classification failed.');
//       }
//     } catch (error) {
//       console.error('Error during classification:', error);
//       Alert.alert('Error', 'Failed to classify link. Please try again.');
//     }
//   };

//   const handleLogout = async () => {
//     try {
//       await AsyncStorage.removeItem('userToken');
//       Alert.alert('Logged Out', 'You have been logged out successfully.');
//       setIsLoggedIn(false);
//     } catch (error) {
//       console.error('Error during logout:', error);
//       Alert.alert('Error', 'An error occurred during logout.');
//     }
//   };

//   return (
//     <ScrollView contentContainerStyle={styles.container}>
//       <Text style={styles.title}>Welcome to News Classifier</Text>
//       <Text style={styles.description}>
//         Classify news articles by providing a link or description.
//       </Text>

//       <View style={styles.featuresContainer}>
//         <Text style={styles.featuresTitle}>
//           Features of the News Classifier App:
//         </Text>
//         {[
//           'Classify Articles: Identify the category of news articles based on the description or link provided.',
//           'Find Similar News: Get recommendations for articles similar to the one you input to stay informed on related topics.',
//           'Bookmark Articles: Save articles to your personal list for easy access later.',
//           'Share News: Share classified articles with others via social media or messaging.',
//           'Detect News: We can also detect the news whether it is fake or not.',
//         ].map((feature, index) => (
//           <View key={index} style={styles.featureItem}>
//             <Text style={styles.bullet}>•</Text>
//             <Text style={styles.featureText}>{feature}</Text>
//           </View>
//         ))}
//       </View>

//       <View style={styles.tabContainer}>
//         <TouchableOpacity
//           style={[styles.tab, activeTab === 'description' && styles.activeTab]}
//           onPress={() => setActiveTab('description')}>
//           <Text
//             style={
//               activeTab === 'description'
//                 ? styles.activeTabText
//                 : styles.tabText
//             }>
//             Description
//           </Text>
//         </TouchableOpacity>
//         <TouchableOpacity
//           style={[styles.tab, activeTab === 'link' && styles.activeTab]}
//           onPress={() => setActiveTab('link')}>
//           <Text
//             style={
//               activeTab === 'link' ? styles.activeTabText : styles.tabText
//             }>
//             Link
//           </Text>
//         </TouchableOpacity>
//       </View>

//       <Card style={styles.card}>
//         <Card.Content>
//           <InputForm
//             inputType={activeTab}
//             value={inputValue}
//             onChangeText={setInputValue}
//           />
//         </Card.Content>
//       </Card>

//       {activeTab === 'description' && (
//         <Button
//           mode="contained"
//           onPress={handleClassifyDescription}
//           style={styles.classifyButton}
//           labelStyle={styles.classifyButtonText}>
//           Classify Description
//         </Button>
//       )}

//       {activeTab === 'link' && (
//         <Button
//           mode="contained"
//           onPress={handleClassifyLink}
//           style={styles.classifyButton}
//           labelStyle={styles.classifyButtonText}>
//           Classify Link
//         </Button>
//       )}
//     </ScrollView>
//   );
// };

// const styles = StyleSheet.create({
//   container: {
//     flexGrow: 1,
//     padding: 20,
//     backgroundColor: '#f2f2f2',
//   },
//   title: {
//     fontSize: 28,
//     fontWeight: 'bold',
//     marginBottom: 10,
//     textAlign: 'center',
//     color: '#2c3e50',
//   },
//   description: {
//     fontSize: 16,
//     color: '#7f8c8d',
//     marginBottom: 20,
//     textAlign: 'center',
//   },
//   featuresContainer: {
//     marginBottom: 20,
//     padding: 15,
//     backgroundColor: '#ecf0f1',
//     borderRadius: 10,
//     elevation: 3,
//   },
//   featuresTitle: {
//     fontSize: 18,
//     fontWeight: 'bold',
//     marginBottom: 10,
//     color: '#34495e',
//   },
//   featureItem: {
//     flexDirection: 'row',
//     alignItems: 'flex-start',
//     marginBottom: 6,
//   },
//   bullet: {
//     fontSize: 18,
//     color: '#3498db',
//     marginRight: 8,
//   },
//   featureText: {
//     fontSize: 15,
//     color: '#7f8c8d',
//     flexShrink: 1,
//   },
//   tabContainer: {
//     flexDirection: 'row',
//     justifyContent: 'center',
//     backgroundColor: '#dfe6e9',
//     borderRadius: 10,
//     overflow: 'hidden',
//     marginBottom: 12,
//   },
//   tab: {
//     flex: 1,
//     paddingVertical: 12,
//     alignItems: 'center',
//   },
//   activeTab: {
//     backgroundColor: '#3498db',
//   },
//   tabText: {
//     fontSize: 16,
//     color: '#7f8c8d',
//   },
//   activeTabText: {
//     fontSize: 16,
//     color: '#fff',
//     fontWeight: 'bold',
//   },
//   card: {
//     backgroundColor: '#ffffff',
//     paddingVertical: 15,
//     paddingHorizontal: 10,
//     borderRadius: 10,
//     elevation: 4,
//     marginBottom: 10,
//   },
//   classifyButton: {
//     marginTop: 5,
//     paddingVertical: 12,
//     borderRadius: 8,
//     backgroundColor: '#3498db',
//     elevation: 2,
//   },
//   classifyButtonText: {
//     fontSize: 18,
//     color: '#ffffff',
//     fontWeight: 'bold',
//   },
//   logoutButton: {
//     marginRight: 15,
//     paddingVertical: 5,
//     paddingHorizontal: 10,
//     backgroundColor: '#e74c3c',
//     borderRadius: 8,
//   },
//   logoutText: {
//     color: '#ffffff',
//     fontSize: 16,
//   },
// });

// export default HomeScreen;
import React, { useState, useLayoutEffect, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Card, Button } from 'react-native-paper';
import { useNavigation } from '@react-navigation/native';
import InputForm from '../components/InputForm';
import ResultModal from '../components/ResultModal';
import axios from 'axios';

const HomeScreen: React.FC<{ setIsLoggedIn: (state: boolean) => void }> = ({
  setIsLoggedIn,
}) => {
  const [activeTab, setActiveTab] = useState<'description' | 'link'>(
    'description'
  );
  const [inputValue, setInputValue] = useState('');
  const [resultModalVisible, setResultModalVisible] = useState(false);
  const [classificationResult, setClassificationResult] = useState({
    category: '',
    // fakeStatus: '',
  });

  const navigation = useNavigation();

  useLayoutEffect(() => {
    navigation.setOptions({
      headerRight: () => (
        <TouchableOpacity onPress={handleLogout} style={styles.logoutButton}>
          <Text style={styles.logoutText}>Logout</Text>
        </TouchableOpacity>
      ),
    });
  }, [navigation]);

  useEffect(() => {
    setInputValue('');
  }, [activeTab]);

  const handleClassifyDescription = async () => {
    if (!inputValue) {
      Alert.alert('Error', 'Please provide a description.');
      return;
    }

    try {
      const response = await axios.post(
        'http://10.38.154.157:3000/api/description',
        {
          description: inputValue,
        }
      );

      if (response.data) {
        const { category } = response.data;
        setClassificationResult({ category });
        setResultModalVisible(true);
      } else {
        Alert.alert('Error', response.data.error || 'Classification failed.');
      }
    } catch (error) {
      console.error('Error during classification:', error);
      Alert.alert('Error', 'Failed to classify description. Please try again.');
    }
  };

  const handleClassifyLink = async () => {
    if (!inputValue) {
      Alert.alert('Error', 'Please provide a link.');
      return;
    }

    try {
      const response = await axios.post('http://10.38.154.157:3000/api/scrape', {
        url: inputValue,
      });

      if (response.data) {
        const { category } = response.data;
        setClassificationResult({ category });
        setResultModalVisible(true);
      } else {
        Alert.alert('Error', response.data.error || 'Classification failed.');
      }
    } catch (error) {
      console.error('Error during classification:', error);
      Alert.alert('Error', 'Failed to classify link. Please try again.');
    }
  };

  const handleLogout = async () => {
    try {
      console.log('Attempting to remove userToken from AsyncStorage');
      await AsyncStorage.removeItem('userToken');
      console.log('userToken removed successfully');
      Alert.alert('Logged Out', 'You have been logged out successfully.');
      navigation.reset({
        index: 0,
        routes: [{ name: 'Login' }],
      }); // Reset navigation stack to prevent going back to logged-in screens
    } catch (error) {
      console.error('Error during logout:', error);
      Alert.alert('Error', 'An error occurred during logout.');
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Welcome to News Classifier</Text>
      <Text style={styles.description}>
        Classify news articles by providing a link or description.
      </Text>

      <View style={styles.featuresContainer}>
        <Text style={styles.featuresTitle}>
          Features of the News Classifier App:
        </Text>
        {[
          'Classify Articles: Identify the category of news articles based on the description or link provided.',
          'Find Similar News: Get recommendations for articles similar to the one you input to stay informed on related topics.',
          'Bookmark Articles: Save articles to your personal list for easy access later.',
          'Share News: Share classified articles with others via social media or messaging.',
          'Detect News: We can also detect the news whether it is fake or not.',
        ].map((feature, index) => (
          <View key={index} style={styles.featureItem}>
            <Text style={styles.bullet}>•</Text>
            <Text style={styles.featureText}>{feature}</Text>
          </View>
        ))}
      </View>

      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'description' && styles.activeTab]}
          onPress={() => setActiveTab('description')}
        >
          <Text
            style={
              activeTab === 'description'
                ? styles.activeTabText
                : styles.tabText
            }
          >
            Description
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'link' && styles.activeTab]}
          onPress={() => setActiveTab('link')}
        >
          <Text
            style={activeTab === 'link' ? styles.activeTabText : styles.tabText}
          >
            Link
          </Text>
        </TouchableOpacity>
      </View>

      <Card style={styles.card}>
        <Card.Content>
          <InputForm
            inputType={activeTab}
            value={inputValue}
            onChangeText={setInputValue}
          />
        </Card.Content>
      </Card>

      {activeTab === 'description' && (
        <Button
          mode="contained"
          onPress={handleClassifyDescription}
          style={styles.classifyButton}
          labelStyle={styles.classifyButtonText}
        >
          Classify Description
        </Button>
      )}

      {activeTab === 'link' && (
        <Button
          mode="contained"
          onPress={handleClassifyLink}
          style={styles.classifyButton}
          labelStyle={styles.classifyButtonText}
        >
          Classify Link
        </Button>
      )}

      <ResultModal
        visible={resultModalVisible}
        onClose={() => setResultModalVisible(false)}
        category={classificationResult.category}
        // fakeStatus={classificationResult.fakeStatus}
      />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 20,
    backgroundColor: '#f2f2f2',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
    color: '#2c3e50',
  },
  description: {
    fontSize: 16,
    color: '#7f8c8d',
    marginBottom: 20,
    textAlign: 'center',
  },
  featuresContainer: {
    marginBottom: 20,
    padding: 15,
    backgroundColor: '#ecf0f1',
    borderRadius: 10,
    elevation: 3,
  },
  featuresTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#34495e',
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 6,
  },
  bullet: {
    fontSize: 18,
    color: '#3498db',
    marginRight: 8,
  },
  featureText: {
    fontSize: 15,
    color: '#7f8c8d',
    flexShrink: 1,
  },
  tabContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    backgroundColor: '#dfe6e9',
    borderRadius: 10,
    overflow: 'hidden',
    marginBottom: 12,
  },
  tab: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
  },
  activeTab: {
    backgroundColor: '#3498db',
  },
  tabText: {
    fontSize: 16,
    color: '#7f8c8d',
  },
  activeTabText: {
    fontSize: 16,
    color: '#fff',
    fontWeight: 'bold',
  },
  card: {
    backgroundColor: '#ffffff',
    paddingVertical: 15,
    paddingHorizontal: 10,
    borderRadius: 10,
    elevation: 4,
    marginBottom: 10,
  },
  classifyButton: {
    marginTop: 5,
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: '#3498db',
    elevation: 2,
  },
  classifyButtonText: {
    fontSize: 18,
    color: '#ffffff',
    fontWeight: 'bold',
  },
  logoutButton: {
    marginRight: 15,
    paddingVertical: 5,
    paddingHorizontal: 10,
    backgroundColor: '#e74c3c',
    borderRadius: 8,
  },
  logoutText: {
    color: '#ffffff',
    fontSize: 16,
  },
  input: {
    color: '#000', // Set text color to black
  },
});

export default HomeScreen;

