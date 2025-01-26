import React from 'react';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import Icon from 'react-native-vector-icons/Ionicons';

import HomeScreen from '../screens/HomeScreen';
import SimilarNewsScreen from '../screens/SimilarNewsScreen.tsx';
import BookmarkedNewsScreen from '../screens/BookmarkedNewsScreen';
import ShareScreen from '../screens/ShareNews';
import TrendingNewsScreen from '../screens/TrendingNewsScreen.tsx';
const Tab = createBottomTabNavigator();

const BottomTabs: React.FC = () => (
  <Tab.Navigator>
    <Tab.Screen
      name="Home"
      component={HomeScreen}
      options={{
        title: 'Home',
        tabBarIcon: ({color, size}) => (
          <Icon name="home-outline" color={color} size={size} />
        ),
      }}
    />
    {/* sssss */}
    <Tab.Screen
      name="Bookmarked News"
      component={BookmarkedNewsScreen}
      options={{
        title: 'Bookmarked News',
        tabBarIcon: ({color, size}) => (
          <Icon name="bookmark-outline" color={color} size={size} />
        ),
      }}
    />
    <Tab.Screen
      name="Share"
      component={ShareScreen}
      options={{
        title: 'Share',
        tabBarIcon: ({color, size}) => (
          <Icon name="share-outline" color={color} size={size} />
        ),
      }}
    />
    <Tab.Screen
      name="Trending News"
      component={TrendingNewsScreen} // Add the Trending News screen
      options={{
        title: 'Trending',
        tabBarIcon: ({color, size}) => (
          <Icon name="flame-outline" color={color} size={size} />
        ),
      }}
    />
  </Tab.Navigator>
);

export default BottomTabs;
