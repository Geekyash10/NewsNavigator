import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, Alert, TouchableOpacity } from 'react-native';
import NewsItemBox from '../components/NewsItemBox';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Icon from 'react-native-vector-icons/Ionicons'; // Import Ionicons

const BOOKMARKS_API = 'http://10.38.154.157:5001/api/bookmarksshown'; // Fetch bookmarks
const REMOVE_BOOKMARK_API = 'http://10.38.154.157:5001/api/removeBookmark'; // Remove bookmark

const SimilarNewsScreen: React.FC = () => {
  const [bookmarkedNews, setBookmarkedNews] = useState([]); // List of bookmarked news
  const [loading, setLoading] = useState(true); // Loading state

  // Fetch bookmarks from the server
  const fetchBookmarks = async () => {
    setLoading(true); // Set loading state during refresh
    try {
      const token = await AsyncStorage.getItem('userToken');
      if (!token) {
        Alert.alert('Error', 'User not authenticated');
        setLoading(false);
        return;
      }

      const response = await axios.get(BOOKMARKS_API, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (response.status === 200) {
        setBookmarkedNews(response.data); // Update bookmarks state
      } else {
        Alert.alert('Error', 'Failed to fetch bookmarks');
      }
    } catch (error) {
      console.error('Error fetching bookmarks:', error);
      Alert.alert('Error', 'Something went wrong while fetching bookmarks.');
    } finally {
      setLoading(false);
    }
  };

  // Remove a bookmark
  const removeBookmark = async (bookmarkId: string) => {
    try {
      const token = await AsyncStorage.getItem('userToken');
      if (!token) {
        Alert.alert('Error', 'User not authenticated');
        return;
      }

      const response = await axios.delete(REMOVE_BOOKMARK_API, {
        headers: { Authorization: `Bearer ${token}` },
        data: { bookmarkId }, // Send bookmark ID in the body
      });

      if (response.status === 200) {
        setBookmarkedNews(response.data.bookmarks); // Update state with remaining bookmarks
        Alert.alert('Success', 'Bookmark removed successfully');
      } else {
        Alert.alert('Error', 'Failed to remove bookmark');
      }
    } catch (error) {
      console.error('Error removing bookmark:', error);
      Alert.alert('Error', 'Something went wrong while removing the bookmark.');
    }
  };

  useEffect(() => {
    fetchBookmarks();
  }, []);

  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#0000ff" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header with title and refresh icon */}
      <View style={styles.header}>
        <Text style={styles.title}>Your Bookmarked News</Text>
        <TouchableOpacity onPress={fetchBookmarks}>
          <Icon name="refresh" size={24} color="#000" style={styles.refreshIcon} />
        </TouchableOpacity>
      </View>

      {bookmarkedNews.length === 0 ? (
        <Text>No bookmarks found</Text>
      ) : (
        bookmarkedNews.map((item) => (
          <NewsItemBox
            key={item._id} // Use unique ID as key
            title={item.title}
            description={item.description}
            url={item.url}
            bookmarkId={item._id} // Pass bookmark ID to the component
            bookmarked={true}
            onRemoveBookmark={removeBookmark} // Pass remove function
          />
        ))
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#fff',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between', // Space between title and icon
    alignItems: 'center',
    marginBottom: 9,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  refreshIcon: {
    marginRight: 8, // Add some padding to the icon
  },
});

export default SimilarNewsScreen;
