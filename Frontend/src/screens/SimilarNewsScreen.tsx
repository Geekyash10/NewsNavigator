// NewsItemBox.tsx
import React from 'react';
import {View, Text, StyleSheet} from 'react-native';

const NewsItemBox: React.FC = () => (
  <>
    <View style={styles.box}>
      <Text style={styles.newsTitle}>News Title</Text>
      <Text style={styles.newsDescription}>
        A brief description or summary of the news item goes here...
      </Text>
    </View>
    <View style={styles.box}>
      <Text style={styles.newsTitle}>News Title</Text>
      <Text style={styles.newsDescription}>
        A brief description or summary of the news item goes here...
      </Text>
    </View>
    <View style={styles.box}>
      <Text style={styles.newsTitle}>News Title</Text>
      <Text style={styles.newsDescription}>
        A brief description or summary of the news item goes here...
      </Text>
    </View>
  </>
);

const styles = StyleSheet.create({
  box: {
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    borderColor: '#ddd',
    borderWidth: 1,
    padding: 16,
    marginBottom: 12,
  },
  newsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  newsDescription: {
    fontSize: 14,
    color: '#666',
  },
});

export default NewsItemBox;
