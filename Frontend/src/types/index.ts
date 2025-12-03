import {BottomTabScreenProps} from '@react-navigation/bottom-tabs';
import {StackScreenProps} from '@react-navigation/stack'; // Used for RootStackParamList
import {RouteProp} from '@react-navigation/native';

// Define the types for each screen in the bottom tab navigator
export type BottomTabParamList = {
  Home: undefined;
  'Similar News': undefined;
  'Bookmarked News': undefined;
};

// Props for each screen in the bottom tab navigator
export type HomeScreenProps = BottomTabScreenProps<BottomTabParamList, 'Home'>;
export type SimilarNewsScreenProps = BottomTabScreenProps<
  BottomTabParamList,
  'Similar News'
>;
export type BookmarkedNewsScreenProps = BottomTabScreenProps<
  BottomTabParamList,
  'Bookmarked News'
>;

// Props that can be passed to components (if any additional props are needed)
export type InputFormProps = {
  onSubmit: (input: string) => void;
};

// Define types for route props if specific params are needed
export type HomeScreenRouteProp = RouteProp<BottomTabParamList, 'Home'>;
export type SimilarNewsScreenRouteProp = RouteProp<
  BottomTabParamList,
  'Similar News'
>;
export type BookmarkedNewsScreenRouteProp = RouteProp<
  BottomTabParamList,
  'Bookmarked News'
>;

// Define the types for the root stack navigator
export type RootStackParamList = {
  Login: undefined;
  Signup: undefined;
  BottomTabs: undefined;
};
