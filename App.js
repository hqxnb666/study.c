// App.js
import React, { useState } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TextInput, 
  TouchableOpacity,
  Alert,
  ScrollView,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform
} from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';
import * as Animatable from 'react-native-animatable';

export default function App() {
  const [inputText, setInputText] = useState('');
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);

  // 预定义的问答集
  const predefinedQA = {
    "你好": "你好呀，很高兴见到你！",
    "今天天气怎么样？": "今天阳光明媚，适合外出哦！",
    "你能做什么？": "我可以陪你聊天、回答问题，还能讲笑话呢！",
    "讲个笑话吧": "为什么程序员喜欢黑色的键盘？因为他们不喜欢白天！",
    "再见": "再见！希望很快再见到你！"
  };

  const handleSend = () => {
    if (!inputText.trim()) {
      Alert.alert('提示', '请输入一点内容哦～');
      return;
    }

    // 添加用户消息到对话记录
    const userMessage = { sender: 'user', text: inputText, time: new Date().toLocaleTimeString() };
    setConversation([...conversation, userMessage]);
    setInputText('');
    setLoading(true);

    // 查找预定义的答案
    const answer = predefinedQA[inputText.trim()];

    if (answer) {
      // 模拟3秒延迟
      setTimeout(() => {
        const aiMessage = { sender: 'ai', text: answer, time: new Date().toLocaleTimeString() };
        setConversation(prev => [...prev, aiMessage]);
        setLoading(false);
      }, 3000);
    } else {
      // 如果问题未预定义，给予默认回复
      setTimeout(() => {
        const aiMessage = { sender: 'ai', text: "抱歉，我还在学习中，无法回答这个问题。", time: new Date().toLocaleTimeString() };
        setConversation(prev => [...prev, aiMessage]);
        setLoading(false);
      }, 3000);
    }
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      keyboardVerticalOffset={90}
    >
      <View style={styles.header}>
        <Text style={styles.title}>简易AI对话示例</Text>
      </View>
      <ScrollView style={styles.chatContainer} contentContainerStyle={{ paddingVertical: 10 }}>
        {conversation.map((msg, index) => (
          <Animatable.View 
            key={index}
            animation="fadeInUp"
            duration={500}
            style={[
              styles.messageBubble, 
              msg.sender === 'user' ? styles.userBubble : styles.aiBubble
            ]}
          >
            <Icon 
              name={msg.sender === 'user' ? 'person-circle' : 'robot-outline'} 
              size={24} 
              color={msg.sender === 'user' ? '#4caf50' : '#555'}
              style={{ marginBottom: 5 }}
            />
            <Text style={styles.messageText}>{msg.text}</Text>
            <Text style={styles.messageTime}>{msg.time}</Text>
          </Animatable.View>
        ))}
        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="small" color="#4caf50" />
            <Text style={styles.loadingText}>AI 正在思考...</Text>
          </View>
        )}
      </ScrollView>
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.textInput}
          placeholder="请输入内容..."
          onChangeText={(text) => setInputText(text)}
          value={inputText}
          onSubmitEditing={handleSend}
          returnKeyType="send"
        />
        <TouchableOpacity style={styles.button} onPress={handleSend}>
          <Icon name="send" size={20} color="#FFF" />
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#EFEFEF',
  },
  header: {
    paddingTop: 50,
    paddingBottom: 10,
    backgroundColor: '#4caf50',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 3,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#fff',
  },
  chatContainer: {
    flex: 1,
    paddingHorizontal: 10,
  },
  messageBubble: {
    padding: 12,
    borderRadius: 15,
    marginVertical: 5,
    maxWidth: '80%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 1,
    elevation: 2,
  },
  userBubble: {
    backgroundColor: '#DCF8C6',
    alignSelf: 'flex-end',
    borderTopRightRadius: 0,
  },
  aiBubble: {
    backgroundColor: '#FFF',
    alignSelf: 'flex-start',
    borderTopLeftRadius: 0,
  },
  messageText: {
    fontSize: 16,
    color: '#333',
  },
  messageTime: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
    textAlign: 'right',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 10,
  },
  loadingText: {
    marginLeft: 10,
    color: '#666',
    fontSize: 14,
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 10,
    backgroundColor: '#FFF',
    alignItems: 'center',
    borderTopWidth: 1,
    borderColor: '#CCC',
  },
  textInput: {
    flex: 1,
    height: 45,
    borderColor: '#CCC',
    borderWidth: 1,
    borderRadius: 25,
    paddingHorizontal: 15,
    backgroundColor: '#F9F9F9',
    marginRight: 10,
  },
  button: {
    backgroundColor: '#4caf50',
    padding: 12,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
  },
  btnText: {
    color: '#FFF',
    fontWeight: '600',
    fontSize: 16,
  },
});
