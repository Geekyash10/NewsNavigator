import React from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  Linking,
} from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';

type TrendingNewsItemProps = {
  title: string;
  source: string;
  author: string | null;
  description: string;
  imageUrl: string | null;
  bookmarked: boolean;
  onToggleBookmark: () => void;
  url: string; // Accept any string URL, not just a specific one
};

const TrendingNewsItem: React.FC<TrendingNewsItemProps> = ({
  title,
  source,
  author,
  description,
  imageUrl,
  bookmarked,
  onToggleBookmark,
  url,
}) => {
  const handleReadMore = () => {
    if (url) {
      Linking.openURL(url);
    } else {
      console.warn('Invalid URL:', url);
    }
  };

  return (
    <View style={styles.container}>
      {imageUrl ? (
        <Image source={{uri: imageUrl}} style={styles.image} />
      ) : null}
      <View style={styles.content}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.source}>
          {source} {author ? `- ${author}` : ''}
        </Text>
        <Text style={styles.description}>
          {description && description.length > 100
            ? `${description.substring(0, 100)}...`
            : description || 'No description available'}
        </Text>

        {/* "Read more" button */}
        <TouchableOpacity onPress={handleReadMore}>
          <Text style={styles.readMore}>Read more</Text>
        </TouchableOpacity>

        {/* Bookmark icon positioned at the bottom right */}
        <TouchableOpacity
          onPress={onToggleBookmark}
          style={styles.bookmarkIcon}>
          <Icon
            name={bookmarked ? 'bookmark' : 'bookmark-outline'}
            size={24}
            color={bookmarked ? '#f39c12' : '#7f8c8d'}
          />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    padding: 10,
    marginBottom: 10,
    backgroundColor: '#ecf0f1',
    borderRadius: 8,
  },
  image: {
    width: 80,
    height: 80,
    borderRadius: 8,
    marginRight: 10,
  },
  content: {
    flex: 1,
    position: 'relative',
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  source: {
    fontSize: 12,
    color: '#7f8c8d',
    marginTop: 4,
  },
  description: {
    fontSize: 14,
    color: '#34495e',
    marginTop: 4,
  },
  readMore: {
    fontSize: 14,
    color: '#2980b9',
    marginTop: 4,
    textDecorationLine: 'underline',
  },
  bookmarkIcon: {
    position: 'absolute',
    bottom: 4,
    right: 4,
  },
});

export default TrendingNewsItem;
