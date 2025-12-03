import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Share, ActivityIndicator, Alert } from 'react-native';
import LottieView from 'lottie-react-native';
import Icon from 'react-native-vector-icons/Ionicons';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const BOOKMARKS_API = 'http://10.38.154.157:5001/api/bookmarksshown';

const ShareScreen: React.FC = () => {
  const [bookmarkedNews, setBookmarkedNews] = useState([]); // State for bookmarks
  const [loading, setLoading] = useState(true); // Loading state

  // Fetch bookmarked news
  const fetchBookmarks = async () => {
    try {
      setLoading(true); // Set loading to true when fetching
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
        setBookmarkedNews(response.data);
      } else {
        Alert.alert('Error', 'Failed to fetch bookmarks');
      }
    } catch (error) {
      console.error('Error fetching bookmarks:', error);
      Alert.alert('Error', 'Something went wrong while fetching bookmarks.');
    } finally {
      setLoading(false); // Stop loading after API call
    }
  };

  useEffect(() => {
    fetchBookmarks();
  }, []);

  // Function to handle sharing
  const onShare = async () => {
    if (bookmarkedNews.length === 0) {
      Alert.alert('No Bookmarks', 'You have no bookmarks to share.');
      return;
    }

    const message = bookmarkedNews
      .map((item, index) => `${index + 1}. ${item.title} - ${item.url}`)
      .join('\n\n');

    try {
      const result = await Share.share({
        message: `Here are some articles I found interesting:\n\n${message}`,
      });

      if (result.action === Share.sharedAction) {
        if (result.activityType) {
          console.log('Shared with activity type:', result.activityType);
        } else {
          console.log('Shared');
        }
      } else if (result.action === Share.dismissedAction) {
        console.log('Share dismissed');
      }
    } catch (error) {
      console.error('Error sharing:', error.message);
    }
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#0000ff" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Refresh Icon */}
      <TouchableOpacity style={styles.refreshIcon} onPress={fetchBookmarks}>
        <Icon name="refresh" size={28} color="#2980b9" />
      </TouchableOpacity>

      {/* Lottie Animation */}
      <LottieView
        source={require('./sharescreen.json')}
        autoPlay
        loop
        style={styles.animation}
      />

      <Text style={styles.title}>Share Your Bookmarked Articles</Text>
      <Text style={styles.description}>
        Share your favorite articles with friends and family!
      </Text>

      {/* Share Button */}
      <TouchableOpacity style={styles.shareButton} onPress={onShare}>
        <Icon name="share-social" size={20} color="#fff" style={styles.icon} />
        <Text style={styles.shareButtonText}>Share Now</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
    backgroundColor: '#f8f9fa',
  },
  refreshIcon: {
    position: 'absolute',
    top: 20,
    right: 20,
  },
  animation: {
    width: 400,
    height: 400,
    marginBottom: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 10,
    textAlign: 'center',
    marginTop: -40,
  },
  description: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
    marginHorizontal: 20,
    marginBottom: 30,
  },
  shareButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 30,
    backgroundColor: '#2980b9',
    borderRadius: 25,
    elevation: 3,
  },
  shareButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  icon: {
    marginRight: 8,
  },
});

export default ShareScreen;
