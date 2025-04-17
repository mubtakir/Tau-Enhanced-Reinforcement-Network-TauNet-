"""
TauNet: نظام تعلم معزز مبتكر يعتمد على مفاهيم رياضية متقدمة
تم تطويره بواسطة: [باسل يحيى عبدالله/ العراق/ الموصل ]
تاريخ الإصدار: [3/4/2025]
"""

"""
TauNet: نظام تعلم معزز متقدم للحفاظ على توازن العمود في بيئة CartPole-v1
المكونات الرئيسية:
1. Dynamic Correlation Layer: تحليل العلاقات الديناميكية بين الميزات.
2. Chaos Optimizer: محسن يعتمد على نظرية الشواش لتحسين الاستكشاف.
3. Hyperbolic Attention: انتباه زائدي لحساب الأهمية النسبية للإجراءات.
آلية العمل:
- تُدخل الحالة الحالية والتاريخ الزمني إلى الشبكة.
- تُحسب قيمة Tau لكل إجراء بناءً على المخاطر والتقدم.
- يُنفذ الإجراء ذو القيمة الأعلى للحفاظ على التوازن.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import deque
import gym
from torch.nn.utils import clip_grad_norm_

# ----------------------- 1. Dynamic Correlation Layer -----------------------
class DynamicCorrelationLayer(nn.Module):
    """
    طبقة الارتباط الديناميكي لمحاكاة التشابك الكمي الكلاسيكي
    تقوم بتحليل العلاقات غير الخطية بين الميزات باستخدام مصفوفة كوفاريانس قابلة للتدريب
    """
    def __init__(self, dim):
        super().__init__()
        self.cov_matrix = nn.Parameter(torch.eye(dim) * 0.1)  # مصفوفة الكوفاريانس
        self.phase = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),  # دالة التنشيط Gaussian Error Linear Unit
            nn.Linear(dim * 2, dim),
            nn.Tanh()
        )
        self.layer_norm = nn.LayerNorm(dim)  # تطبيع الطبقة لتحقيق الاستقرار
        
    def forward(self, x):
        """تمرير البيانات عبر الطبقة"""
        cov = torch.matmul(x, self.cov_matrix)  # عملية الارتباط
        phase_out = self.phase(cov)  # تعديل الطور
        return self.layer_norm(phase_out + x)  # وصلة التخطي

# ----------------------- 2. Chaos Optimizer -----------------------
class ChaosOptimizer(optim.Optimizer):
    """
    محسن يعتمد على نظرية الشواش باستخدام معادلات لورنز المعدلة
    يحسن عملية الاستكشاف في الفضاء الحلولي
    """
    def __init__(self, params, lr=1e-3, sigma=10.0, rho=28.0, beta=8/3):
        defaults = dict(lr=lr, sigma=sigma, rho=rho, beta=beta)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        """خطوة تحديث المعلمات باستخدام معادلات الشواش"""
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if 'chaos_state' not in state:
                    state['chaos_state'] = torch.randn(3, device=grad.device)
                
                x, y, z = state['chaos_state']
                
                # معادلات لورنز المعدلة
                dx = group['sigma'] * (y - x)
                dy = x * (group['rho'] - z) - y
                dz = x * y - group['beta'] * z
                
                delta = torch.stack([dx, dy, dz]).mean() * group['lr']
                p.data.add_(delta)
                
                state['chaos_state'] = torch.tensor([dx, dy, dz], device=grad.device)
                
        return loss

# ----------------------- 3. Hyperbolic Attention -----------------------
class HyperbolicAttention(nn.Module):
    """
    طبقة الانتباه الزائدي لحساب الأهمية النسبية في الفضاء الزائدي
    تعتمد على هندسة لوباتشيفسكي للتعامل مع العلاقات الهرمية
    """
    def __init__(self, dim, c=0.01):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(c))  # انحناء الفضاء الزائدي
        self.scale = math.sqrt(dim)
        self.eps = 1e-6  # قيمة صغيرة لمنع القسمة على الصفر
        
    def hyperbolic_distance(self, x, y):
        """حساب المسافة الزائدية بين النقاط"""
        batch, seq_x, dim = x.size()
        seq_y = y.size(1)
        
        # توسيع الأبعاد للحساب الزوجي
        x_expanded = x.unsqueeze(2)  # (batch, seq_x, 1, dim)
        y_expanded = y.unsqueeze(1)  # (batch, 1, seq_y, dim)
        
        diff = x_expanded - y_expanded
        norm_diff = torch.norm(diff, p=2, dim=-1, keepdim=False)
        
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
        
        denominator = (1 - self.c * x_norm**2) * (1 - self.c * y_norm**2)
        denominator = torch.clamp(denominator, min=self.eps)
        
        argument = 1 + 2 * self.c * (norm_diff**2) / denominator
        argument = torch.clamp(argument, min=1 + self.eps)
        
        return torch.acosh(argument)
    
    def forward(self, query, keys, values):
        """عملية الانتباه الزائدي"""
        dist_matrix = self.hyperbolic_distance(query, keys)
        attn_weights = torch.softmax(-dist_matrix / self.scale, dim=-1)
        return torch.bmm(attn_weights, values)  # تجميع القيم المرجحة

# ----------------------- 4. TauNet Architecture -----------------------
class TauNet(nn.Module):
    """
    البنية الرئيسية لشبكة TauNet
    تدمج المكونات السابقة لاتخاذ قرارات مثلى في البيئات الديناميكية
    """
    def __init__(self, input_dim, hidden_dim=128, num_actions=2):
        super().__init__()
        # طبقة إسقاط المدخلات
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU()
        )
        
        # التشفير الزمني باستخدام LSTM ثنائي الاتجاه
        self.temporal_enc = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # مكونات النظام
        self.correlation = DynamicCorrelationLayer(hidden_dim * 2)
        self.attention = HyperbolicAttention(hidden_dim * 2)
        
        # رأس تقدير المخاطرة
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_actions),
            nn.Softplus()  # ضمان قيم موجبة
        )
        
        # رأس تقدير التقدم
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),  # إخراج بين -1 و1
            nn.Linear(hidden_dim, num_actions)
        )
        
        # معلمات تكيفية لمعادلة Tau
        self.adaptive_scale = nn.Parameter(torch.ones(num_actions))
        self.adaptive_bias = nn.Parameter(torch.zeros(num_actions))
        
    def forward(self, x, history):
        """التدفق الأمامي للشبكة"""
        # معالجة المدخلات الحالية
        x_proj = self.input_proj(x)
        
        # التشفير الزمني للتاريخ
        temporal_out, _ = self.temporal_enc(history)
        temporal_feat = temporal_out.mean(dim=1)
        
        # الارتباط الديناميكي
        correlated = self.correlation(temporal_feat)
        
        # الانتباه الزائدي
        attn_out = self.attention(
            x_proj.unsqueeze(1),
            correlated.unsqueeze(1),
            correlated.unsqueeze(1)
        ).squeeze(1)
        
        # حساب المخاطرة والتقدم
        risk = self.risk_head(attn_out)
        progress = self.progress_head(attn_out)
        
        # معادلة Tau الأساسية
        numerator = progress * torch.tanh(risk * self.adaptive_scale + self.adaptive_bias)
        denominator = risk + 1e-6 * (1 + torch.sigmoid(progress))
        tau = numerator / (denominator + 1e-6)
        
        return tau, risk

# ----------------------- 5. نظام التعلم المعزز -----------------------
class TauRL:
    """
    نظام التعلم المعزز الكامل باستخدام TauNet
    يدير عملية التدريب والتفاعل مع البيئة
    """
    def __init__(self, env, hidden_dim=128, buffer_size=20000):
        self.env = env
        self.num_actions = env.action_space.n  # عدد الإجراءات الممكنة
        obs_dim = env.observation_space.shape[0]
        
        # تهيئة الشبكة
        self.net = TauNet(obs_dim, hidden_dim, self.num_actions)
        self.optimizer = ChaosOptimizer(self.net.parameters(), lr=1e-3)
        
        # ذاكرة الخبرة
        self.buffer = deque(maxlen=buffer_size)
        self.history = deque(maxlen=50)  # نافذة زمنية للتاريخ
        
        # إحصاءات المكافآت
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 1e-4
        
    def remember(self, state, action, reward, next_state, done):
        """تخزين الخبرة في الذاكرة"""
        self.history.append(state)
        normalized_reward = (reward - self.reward_mean) / (self.reward_std + 1e-6)
        self.buffer.append((
            state.copy(),
            list(self.history),
            action,
            normalized_reward,
            next_state.copy(),
            done
        ))
        
        # تحديث الإحصاءات
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std += (delta * delta2 - self.reward_std) / self.reward_count
        
    def _process_histories(self, histories):
        """معالجة التاريخ الزمني مع Padding"""
        max_len = max(len(h) for h in histories)
        padded = []
        for h in histories:
            h_array = np.array(h[-50:])
            if len(h_array) < max_len:
                pad = np.zeros((max_len - len(h_array), h_array.shape[1]))
                h_array = np.concatenate([h_array, pad], axis=0)
            padded.append(torch.FloatTensor(h_array))
        return torch.stack(padded)
    
    def train_step(self, batch_size=256):
        """خطوة تدريب واحدة"""
        if len(self.buffer) < batch_size:
            return None
        
        # عينة عشوائية من الذاكرة
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # تفريغ الدفعة
        states, histories, actions, rewards, next_states, dones = zip(*batch)
        
        # تحويل إلى تنسورات
        states_tensor = torch.FloatTensor(np.array(states))
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)
        
        # معالجة التاريخ
        history_tensor = self._process_histories(histories)
        
        # حساب قيم Q الحالية
        current_tau, _ = self.net(states_tensor, history_tensor)
        current_q = current_tau.gather(1, actions_tensor.unsqueeze(1))
        
        # حساب قيم Q الهدف
        with torch.no_grad():
            next_tau, _ = self.net(next_states_tensor, history_tensor)
            target_q = rewards_tensor + 0.99 * next_tau.max(1)[0] * (1 - dones_tensor)
        
        # حساب الخسارة
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
        
        # تحديث المعلمات
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, episodes=500):
        """عملية التدريب الرئيسية"""
        best_reward = -np.inf
        for ep in range(episodes):
            # إعادة تعيين البيئة
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            self.history = deque([state] * 3, maxlen=50)
            total_reward = 0
            done = False
            
            while not done:
                # استراتيجية استكشاف ε-greedy
                epsilon = 0.1 + 0.9 * math.exp(-ep / 100)
                if np.random.rand() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        history_tensor = self._process_histories([list(self.history)])
                        tau, _ = self.net(state_tensor, history_tensor)
                        action = tau.argmax().item()
                
                # تنفيذ الإجراء
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # حفظ الخبرة
                self.remember(state, action, reward, next_state, done)
                self.history.append(next_state)
                state = next_state.copy()
                
                # تدريب دوري
                if len(self.buffer) >= 256 and ep % 4 == 0:
                    self.train_step()
            
            # تسجيل النتائج
            print(f"الحلقة {ep+1}/{episodes}, المكافأة: {total_reward:.1f}")
            
            # حفظ أفضل نموذج
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.net.state_dict(), "taunet_best.pth")

# ----------------------- التنفيذ الرئيسي -----------------------
if __name__ == "__main__":
    # تهيئة البيئة
    env = gym.make('CartPole-v1', render_mode='human')
    
    # شرح البيئة:
    # البيئة تتكون من عربة تتحرك على قضيب أفقي مع عمود متصل بها.
    # الهدف هو إبقاء العمود في وضع مستقيم عن طريق تحريك العربة يمينًا أو يسارًا.
    # المشكلة التي تحلها الذراع/المدفعية: الحفاظ على توازن العمود.
    # ما تتجنبه: سقوط العمود عن الوضع الرأسي بأكثر من 15 درجة.
    
    # تهيئة الوكيل
    agent = TauRL(env)
    
    # بدء التدريب
    agent.train(episodes=500)