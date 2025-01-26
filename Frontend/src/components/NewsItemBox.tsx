import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Linking } from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';

const NewsItemBox: React.FC<{
  title: string;
  description?: string;
  url: string;
  bookmarkId: string; // Unique bookmark ID
  bookmarked: boolean;
  onRemoveBookmark: (bookmarkId: string) => void; // Callback for removing bookmark
}> = ({ title, description, url, bookmarkId, bookmarked, onRemoveBookmark }) => {
  const openUrl = async () => {
    try {
      await Linking.openURL(url);
    } catch (error) {
      console.error('Failed to open URL:', error);
    }
  };

  return (
    <View style={styles.box}>
      <Text style={styles.title}>{title}</Text>
      {description && <Text style={styles.description}>{description}</Text>}

      <TouchableOpacity onPress={openUrl}>
        <Text style={styles.url}>{url}</Text>
      </TouchableOpacity>

      {/* Bookmark icon with remove functionality */}
      <TouchableOpacity
        style={styles.bookmarkIconContainer}
        onPress={() => onRemoveBookmark(bookmarkId)}
      >
        <Icon name="bookmark" size={24} color="#FFD700" />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  box: {
    padding: 16,
    borderBottomWidth: 1,
    borderColor: '#ccc',
    marginBottom: 8,
    position: 'relative',
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  description: {
    fontSize: 14,
    color: '#666',
  },
  url: {
    marginTop: 8,
    color: '#1e90ff',
    textDecorationLine: 'underline',
  },
  bookmarkIconContainer: {
    position: 'absolute',
    bottom: 10,
    right: 10,
  },
});

export default NewsItemBox;
