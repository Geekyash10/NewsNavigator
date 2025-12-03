// import React, { useState, useEffect } from 'react';
// import {
//   View,
//   Text,
//   FlatList,
//   StyleSheet,
//   ActivityIndicator,
//   Alert,
//   RefreshControl,
//   TouchableOpacity,
// } from 'react-native';
// import LottieView from 'lottie-react-native';
// import TrendingNewsItem from '../components/Trendingnews';
// import axios from 'axios';
// import AsyncStorage from '@react-native-async-storage/async-storage';
// import Icon from 'react-native-vector-icons/Ionicons';
// import fallbackData from './trending_news.json'; // Import local JSON fallback data

// const API_URL = 'https://newsdata.io/api/1/news';
// const API_KEY = 'pub_589950ec57a4697dc0572b98f392550e505d9'; // Your API key
// const BOOKMARK_API = 'http://10.38.154.157:5000/bookmark'; // Your server URL for bookmarks

// const TrendingNewsScreen: React.FC = () => {
//   const [newsList, setNewsList] = useState([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);
//   const [refreshing, setRefreshing] = useState(false);

//   useEffect(() => {
//     fetchTrendingNews();
//   }, []);

//   const fetchTrendingNews = async () => {
//     try {
//       setLoading(true);
//       setError(null);

//       const response = await fetch(`${API_URL}?apikey=${API_KEY}&q=india`);

//       if (!response.ok) {
//         throw new Error(`HTTP error! Status: ${response.status}`);
//       }

//       const data = await response.json();

//       if (data.results) {
//         setNewsList(
//           data.results.map((article, index) => ({
//             id: article.article_id || index.toString(),
//             title: article.title || 'No title available',
//             source: article.source_name || 'Unknown source',
//             author: article.creator ? article.creator[0] : 'Unknown',
//             description: article.description || 'No description available',
//             url: article.link,
//             bookmarked: false,
//           })),
//         );
//       } else {
//         setError('No articles found.');
//       }
//     } catch (error) {
//       console.error('Error fetching news:', error);

//       // Use local fallback data in case of an error
//       setError(
//         error.message === 'Network request failed'
//           ? 'Network error! Showing offline data.'
//           : 'Failed to fetch news. Showing offline data.',
//       );
//       setNewsList(
//         fallbackData.newsList.map((item, index) => ({
//           ...item,
//           id: item.id || index.toString(), // Ensure IDs are consistent
//         })),
//       );
//     } finally {
//       setLoading(false);
//       setRefreshing(false);
//     }
//   };

//   const toggleBookmark = async (newsItem: any) => {
//     const isBookmarked = newsList.find(
//       item => item.id === newsItem.id,
//     )?.bookmarked;

//     if (!isBookmarked) {
//       try {
//         const token = await AsyncStorage.getItem('userToken');
//         if (!token) {
//           Alert.alert('Error', 'User not authenticated');
//           return;
//         }

//         // Send only necessary fields
//         const bookmarkData = {
//           title: newsItem.title,
//           description: newsItem.description,
//           url: newsItem.url,
//         };

//         const response = await axios.post(BOOKMARK_API, bookmarkData, {
//           headers: {
//             Authorization: `Bearer ${token}`,
//           },
//         });

//         if (response.status === 200) {
//           setNewsList(prevList =>
//             prevList.map(item =>
//               item.id === newsItem.id ? { ...item, bookmarked: true } : item,
//             ),
//           );
//           Alert.alert('Success', 'Item bookmarked!');
//         }
//       } catch (error) {
//         Alert.alert('Error', 'Failed to add bookmark.');
//         console.error('Error adding bookmark:', error.response?.data || error);
//       }
//     } else {
//       Alert.alert('Notice', 'This item is already bookmarked.');
//     }
//   };

//   const handleRefresh = () => {
//     setRefreshing(true);
//     fetchTrendingNews();
//   };

//   return (
//     <View style={styles.container}>
//       <View style={styles.header}>
//         <Text style={styles.title}>Trending News</Text>
//         <TouchableOpacity onPress={handleRefresh}>
//           <Icon name="refresh" size={28} color="#2c3e50" />
//         </TouchableOpacity>
//       </View>
//       {loading ? (
//         <ActivityIndicator size="large" color="#3498db" style={styles.loading} />
//       ) : error ? (
//         <View style={styles.errorContainer}>
//           <LottieView
//             source={require('../screens/networkerror.json')}
//             autoPlay
//             loop
//             style={styles.errorAnimation}
//           />
//           <Text style={styles.errorText}>{error}</Text>
//         </View>
//       ) : (
//         <FlatList
//           data={newsList}
//           keyExtractor={item => item.id}
//           renderItem={({ item }) => (
//             <TrendingNewsItem
//               title={item.title}
//               source={item.source}
//               author={item.author}
//               description={item.description}
//               bookmarked={item.bookmarked}
//               onToggleBookmark={() => toggleBookmark(item)}
//             />
//           )}
//           ListEmptyComponent={
//             <Text style={styles.emptyText}>No news articles available.</Text>
//           }
//           contentContainerStyle={
//             newsList.length === 0 ? styles.centerContent : {}
//           }
//           refreshControl={
//             <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
//           }
//         />
//       )}
//     </View>
//   );
// };

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     padding: 16,
//     backgroundColor: '#f2f2f2',
//   },
//   header: {
//     flexDirection: 'row',
//     justifyContent: 'space-between',
//    alignItems:'center',
//     marginBottom: 16,
//   },
//   title: {
//     fontSize: 26,
//     fontWeight: 'bold',
//     marginBottom: 16,
//     color: '#2c3e50',
//     textAlign: 'center',
//   },
//   loading: {
//     marginTop: 10,
//   },
//   errorContainer: {
//     flex: 1,
//     justifyContent: 'center',
//     marginTop: -40,
//   },
//   errorAnimation: {
//     width: 400,
//     height: 500,
//   },
//   errorText: {
//     color: 'red',
//     fontSize: 32,
//     textAlign: 'center',
//     marginTop: 10,
//     fontWeight: '800',
//   },
//   emptyText: {
//     textAlign: 'center',
//     color: '#7f8c8d',
//     fontSize: 20,
//     marginVertical: 20,
//   },
//   centerContent: {
//     flexGrow: 1,
//     justifyContent: 'center',
//   },
// });

// export default TrendingNewsScreen;


import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  Alert,
  RefreshControl,
  TouchableOpacity,
} from 'react-native';
import LottieView from 'lottie-react-native';
import TrendingNewsItem from '../components/Trendingnews';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Icon from 'react-native-vector-icons/Ionicons';
import fallbackData from './trending_news.json'; // Import local JSON data
import axios from 'axios';

const BOOKMARK_API = 'http://10.38.154.157:5001/api/bookmark'; // Your server URL for bookmarks

const TrendingNewsScreen: React.FC = () => {
  const [newsList, setNewsList] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadOfflineNews();
  }, []);

  const loadOfflineNews = () => {
    // Load the local JSON data
    setNewsList(
      fallbackData.newsList.map((item, index) => ({
        ...item,
        id: item.id || index.toString(), // Ensure IDs are consistent
      })),
    );
  };

  const toggleBookmark = async (newsItem: any) => {
    const isBookmarked = newsList.find(item => item.id === newsItem.id)?.bookmarked;
  
    if (!isBookmarked) {
      try {
        const token = await AsyncStorage.getItem('userToken');
        if (!token) {
          Alert.alert('Error', 'User not authenticated');
          return;
        }
  
        const bookmarkData = {
          title: newsItem.title,
          description: newsItem.description,
          url: newsItem.url,
        };
  
        console.log('Bookmark Data:', bookmarkData);
  
        // Using axios for POST request
        const response = await axios.post(BOOKMARK_API, bookmarkData, {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });
  
        console.log('Response:', response.data);
  
        if (response.status === 200) {
          setNewsList(prevList =>
            prevList.map(item =>
              item.id === newsItem.id ? { ...item, bookmarked: true } : item,
            ),
          );
          Alert.alert('Success', 'Item bookmarked!');
        } else {
          Alert.alert('Error', response.data.message || 'Failed to bookmark the item.');
        }
      } catch (error: any) {
        console.error('Error adding bookmark:', error.response?.data || error.message);
        Alert.alert('Error', error.response?.data?.message || 'Failed to add bookmark.');
      }
    } else {
      Alert.alert('Notice', 'This item is already bookmarked.');
    }
  };
  const handleRefresh = () => {
    setRefreshing(true);
    loadOfflineNews();
    setRefreshing(false);
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Trending News</Text>
        <TouchableOpacity onPress={handleRefresh}>
          <Icon name="refresh" size={28} color="#2c3e50" />
        </TouchableOpacity>
      </View>
      <FlatList
        data={newsList}
        keyExtractor={item => item.id}
        renderItem={({ item }) => (
          <TrendingNewsItem
            title={item.title}
            source={item.source}
            author={item.author}
            description={item.description}
            bookmarked={item.bookmarked}
            onToggleBookmark={() => toggleBookmark(item)}
          />
        )}
        ListEmptyComponent={
          <Text style={styles.emptyText}>No news articles available.</Text>
        }
        contentContainerStyle={
          newsList.length === 0 ? styles.centerContent : {}
        }
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#f2f2f2',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#2c3e50',
    textAlign: 'center',
  },
  emptyText: {
    textAlign: 'center',
    color: '#7f8c8d',
    fontSize: 20,
    marginVertical: 20,
  },
  centerContent: {
    flexGrow: 1,
    justifyContent: 'center',
  },
});

export default TrendingNewsScreen;
