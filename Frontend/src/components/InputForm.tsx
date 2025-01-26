import React from 'react';
import {View, TextInput, StyleSheet} from 'react-native';

interface InputFormProps {
  inputType: 'description' | 'link';
  value: string;
  onChangeText: (text: string) => void;
}

const InputForm: React.FC<InputFormProps> = ({
  inputType,
  value,
  onChangeText,
}) => {
  const placeholderText =
    inputType === 'description'
      ? 'Enter a news description...'
      : 'Enter a news article link...';

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder={placeholderText}
        value={value}
        onChangeText={onChangeText}
        multiline={inputType === 'description'}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    backgroundColor: '#fff',
  },
});

export default InputForm;
