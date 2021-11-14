import paddle
from paddle import nn

# Paddle Layer
class PatchEmbed(nn.Layer):
    """嵌入层 -- 图像的输入到Patch数据的映射生成
        Params Infos:
            img_size:    输入图像大小
            in_channels: 输入通道大小
            patch_size:  划分的patch大小
            embed_dims:  要嵌入的维度的大小
        Forward Tips:
            - Image To Patch Sequence
            - Patch Size Embed to D Feature Size
            - Concat Classifer embed
            - Add Position embed
    """
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 patch_size=16,
                 embed_dims=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size # p*p
        self.embed_dims = embed_dims
        # H*W -- H**2
        self.num_patch = (img_size // patch_size) ** 2 # N

        # P*P*C
        self.linear = nn.Linear(in_features=patch_size**2 * in_channels,
                                out_features=embed_dims)

        # Position Embeddings -- 1, N+1 ,D
        self.position_embed = self.create_parameter(
            shape=[1, self.num_patch+1, embed_dims], dtype='float32')
        
        # Classifier Embeddings  -- 1, 1, D
        self.classifier_embed = self.create_parameter(
            shape=[1, 1, embed_dims], dtype='float32')

    def forward(self, inputs):
        """二维映射说明
            kernel_size = patch_size,
            stride = patch_size,
            in_channels(3) --> out_channels==embed_dims(768)
        """

        # B, C, H, W
        B, C, H, W = inputs.shape
        x = inputs.transpose(perm=[0, 2, 3, 1]) # B, H, W, C
        # B, H//P, P, W//P, P, C
        x = x.reshape(shape=[B, H//self.patch_size, self.patch_size
                          , W//self.patch_size, self.patch_size, C])
        x = x.transpose(perm=[0, 1, 3, 2, 4, 5]) # B, H//P, W//P, P, P, C
        # B, N, P*P*C
        x = x.reshape(shape=[B, H//self.patch_size*W//self.patch_size,
                             self.patch_size*self.patch_size*C])

        x = self.linear(x) # B, N, D(embeding size)
        # 归一化操作 -- no do

        # print('Add CLas Token Before: ', x.shape)

        # cls_token -- 1, 1, D
        # patch_embed -- B, N, D
        x = paddle.concat([self.classifier_embed, x],
                          axis=1)
        # print('Add CLas Token After: ', x.shape)
        """
        Add CLas Token Before:  [1, 196, 768]
        Add CLas Token After:  [1, 197, 768]
        """

        x = x + self.position_embed

        return x


class MLP(nn.Layer):
    """多层感知机
        Params Info:
            in_features: 输入特征大小
            out_features: 输出特征大小，default:None
            mlp_ratio: MLP中隐藏层伸缩比例，in_features*mlp_ratio
            dropout_rate: 丢弃率
            act: 激活函数 -- nn.Layer or nn.functional
        Forward Tips:
            - 将输入特征映射到更高维去学习隐藏特征
            - 然后经过激活，丢弃 --> 再回到原始输入特征大小
    """
    def __init__(self,
                 in_features,
                 out_features=None,
                 mlp_ratio=4,
                 dropout_rate=0.,
                 act=nn.GELU):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = in_features if out_features is None \
                            else out_features
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate

        # 将输入映射到隐藏特征维度
        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=int(in_features * mlp_ratio))
        # 将输入从隐藏特征维度降回指定的输出维度
        self.fc2 = nn.Linear(in_features=int(in_features * mlp_ratio),
                             out_features=self.out_features)

        self.act = act()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, inputs):
        x = self.fc1(inputs) # Low to High
        x = self.act(x)
        x = self.dropout(x)

        # print("隐藏层大小:", x.shape)
        """
        隐藏层大小: [1, 197, 3072]
        MLP还原后的大小: [1, 197, 768]
        """

        x = self.fc2(x) # High to Low
        x = self.dropout(x)

        # print("MLP还原后的大小:", x.shape)

        return x


class Attention(nn.Layer):
    """注意力机制的实现 -- 伸缩点积模型
        Params Info:
            embed_dims: 嵌入维度大小
            num_head: 注意力头数
            attn_dropout_rate: 注意力分布的丢弃率
            dropout_rate: 注意力结果的丢弃率
        Forward Tips:
            - 输入Patch数据，映射QKV矩阵
            - 再将QKV矩阵转换为多头的数据形式，保证头数与patch数维度要交换
            - 将其计算一般的注意力步骤，并获取多头结果
            - 将多头的结果进行组合还原 --> 通过一个线性层保证输入大小的嵌入维度
    """
    def __init__(self,
                 embed_dims=768,
                 num_head=12,
                 attn_dropout_rate=0.,
                 dropout_rate=0.):
        super(Attention, self).__init__()
        self.embed_dims = embed_dims
        self.num_head = num_head
        self.attn_dropout_rate = attn_dropout_rate
        self.dropout_rate = dropout_rate
        assert embed_dims % num_head == 0, \
            "Warning: Please make sure the embed_dims % num_head == 0,"+\
            "but embed_dims={0}, num_head={0}.".format(embed_dims, num_head)
        
        # embed_dims % num_head != 0 ... 1
        # 24, 6 --> 24
        # 72
        # 24 --> 24
        
        self.head_dims = embed_dims // num_head
        self.scale = self.head_dims ** -0.5

        # B, N+1, D --> B, N+1, 3*D
        self.qkv_proj = nn.Linear(in_features=embed_dims,
                                  out_features=3*self.head_dims*self.num_head)
        self.out = nn.Linear(in_features=self.head_dims*self.num_head,
                             out_features=embed_dims)

        self.softmax = nn.Softmax()
        self.attn_dropout = nn.Dropout(p=attn_dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # inputs: B, N, D
        qkv = self.qkv_proj(inputs) # B, N, 3*D
        # print("QKV集体映射大小: ", qkv.shape)
        q, k, v = qkv.chunk(3, axis=-1) # B, N, D
        # print("QKV划分为单个参数矩阵: ", q.shape)

        B, N, D = q.shape
        # D // self.head_dims,  self.head_dims
        # B, N, num_head, head_dims
        q = q.reshape(shape=[B, N, self.num_head, self.head_dims])
        # tranpose => B, num_head, N, head_dims
        q = q.transpose(perm=[0, 2, 1, 3])
        # print("Query变换后数据形状: ", q.shape)
        k = k.reshape(shape=[B, N, self.num_head, self.head_dims])
        k = k.transpose(perm=[0, 2, 1, 3])
        v = v.reshape(shape=[B, N, self.num_head, self.head_dims])
        v = v.transpose(perm=[0, 2, 1, 3])

        attn = paddle.matmul(q, k, transpose_y=True) # B, num_head, N, N
        attn = attn * self.scale
        attn = self.softmax(attn)  # 注意力分布
        attn = self.attn_dropout(attn)
        # print("注意力分布数据形状: ", attn.shape)

        z = paddle.matmul(attn, v) # B, num_head, N, head_dims
        # print("注意力结果数据形状: ", z.shape)

        z = z.transpose(perm=[0, 2, 1, 3]) # B, N, num_head, head_dims
        z = z.reshape(shape=[B, N, self.num_head*self.head_dims]) # B,N,D
        z = self.out(z) # B, N, embed_size(D)
        z = self.dropout(z)
        # print("注意力结果映射后的数据形状: ", z.shape)

        """
        QKV集体映射大小:  [1, 197, 2304] # 768*3
        QKV划分为单个参数矩阵:  [1, 197, 768]
        Query变换后数据形状:  [1, 12, 197, 64] # 197,64 -- 64,197
        注意力分布数据形状:  [1, 12, 197, 197] # 197,197 -- 197, 64
        注意力结果数据形状:  [1, 12, 197, 64] # 197, 64
        注意力结果映射后的数据形状:  [1, 197, 768]
        隐藏层大小: [1, 197, 3072]
        MLP还原后的大小: [1, 197, 768]
        """

        return z


class DropPath(nn.Layer):
    """多分支的Dropout -- B, N, C --> 沿着B这个分支维度进行丢弃
    """
    def __init__(self, p=0.):
        super(DropPath, self).__init__()
        self.p = p # 0.3
    
    def forward(self, inputs):
        if self.p > 0. and self.training:
            keep_p = 1 - self.p # 0.7
            keep_p = paddle.to_tensor([keep_p], dtype='float32')
            # B, 1, 1
            # [B] + [1]*(inputs.ndim-1) == [1, 1]
            # [B, 1, 1]
            shape = [inputs.shape[0]] + [1.]*(inputs.ndim - 1)
            # 0.7 + [0., 1.]
            random_keep = keep_p + paddle.rand(shape=shape, dtype='float32')
            # > 1.0 == 1
            # < 1.0 == 0
            random_mask = random_keep.floor() # 向下取整
            # inputs: B, N ,D
            # random_mask: B, 1, 1
            # 1, N, D --> 都被丢弃
            output = inputs.divide(keep_p) * random_mask # 保持总的期望不变

            return output
        
        return inputs


class EncoderLayer(nn.Layer):
    """VIT中的基本组件
        Params Info:
            embed_dims: 嵌入维度大小
            mlp_ratio: MLP隐藏层伸缩比例
            num_head: 注意力头数
            attn_dropout_rate: 注意力分布丢弃率
            dropout_rate: 注意力结果丢弃率以及MLP中神经元丢弃率
            droppath_rate: 多分支丢弃率
            act: 激活函数
            norm: 归一化层
        Forward Tips:
            - 输入Patch数据，经过归一化，再通过多头注意力处理
            - 再一次经过归一化，通过MLP处理，得到结果
            - 输入大小==输出大小
    """
    def __init__(self,
                 embed_dims=768,
                 mlp_ratio=4,
                 num_head=12,
                 attn_dropout_rate=0.,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 act=nn.GELU,
                 norm=nn.LayerNorm):
        super(EncoderLayer, self).__init__()
        self.embed_dims = embed_dims
        self.mlp_ratio = mlp_ratio
        self.num_head = num_head
        self.attn_dropout_rate = attn_dropout_rate
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate

        self.attn_norm = norm(embed_dims)
        self.mlp_norm = norm(embed_dims)

        self.multi_attn = Attention(embed_dims=embed_dims,
                                    num_head=num_head,
                                    attn_dropout_rate=attn_dropout_rate,
                                    dropout_rate=dropout_rate)
        
        self.mlp = MLP(in_features=embed_dims,
                       mlp_ratio=mlp_ratio,
                       dropout_rate=dropout_rate,
                       act=act)

        self.attn_droppath = DropPath(p=droppath_rate)
        self.mlp_droppath = DropPath(p=droppath_rate)
    
    def forward(self, inputs):
        res = inputs
        x = self.attn_norm(inputs)
        x = self.multi_attn(x) # B, N, D
        x = self.attn_droppath(x)
        x = x + res

        res = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.mlp_droppath(x)
        x = x + res

        return x


class Encoder(nn.Layer):
    """Transformer Encoder -- 输入大小等于输出大小
        Params Infos:
            num_layers: Encoder中堆叠的层数
            embed_dims: 嵌入大小
            mlp_ratio: MLP隐藏层伸缩比例
            num_head: 注意力头数
            attn_dropout_rate: 注意力分布丢弃率
            dropout_rate: MLP神经元与注意力结果丢弃率
            droppath_rate: 多分支丢弃率
            act: 激活函数
            norm: 归一化层
        Forward Tips:
            - 串联层的操作 -- 顺序层
    """
    def __init__(self,
                 num_layers=12,
                 embed_dims=768,
                 mlp_ratio=4,
                 num_head=12,
                 attn_dropout_rate=0.,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 act=nn.GELU,
                 norm=nn.LayerNorm):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.mlp_ratio = mlp_ratio
        self.num_head = num_head
        self.attn_dropout_rate = attn_dropout_rate
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate

        blocks = []
        for i in range(num_layers):
            blocks.append(
                EncoderLayer(
                    embed_dims=embed_dims,
                    mlp_ratio=mlp_ratio,
                    num_head=num_head,
                    attn_dropout_rate=attn_dropout_rate,
                    dropout_rate=dropout_rate,
                    droppath_rate=droppath_rate,
                    act=act,
                    norm=norm
                )
            )
        self.encoder_blocks = nn.LayerList(blocks) # 像list一样可以索引
        # self.encoder_blocks = nn.Sequential(*blocks)

    def forward(self, inputs):
        x = inputs

        for i in range(0, self.num_layers):
            x = self.encoder_blocks[i](x)
            # print("第 {0} Layer Shape: {1}".format(i, x.shape))

            """
            第 0 Layer Shape: [1, 197, 768]
            第 1 Layer Shape: [1, 197, 768]
            第 2 Layer Shape: [1, 197, 768]
            第 3 Layer Shape: [1, 197, 768]
            第 4 Layer Shape: [1, 197, 768]
            第 5 Layer Shape: [1, 197, 768]
            第 6 Layer Shape: [1, 197, 768]
            第 7 Layer Shape: [1, 197, 768]
            第 8 Layer Shape: [1, 197, 768]
            第 9 Layer Shape: [1, 197, 768]
            第 10 Layer Shape: [1, 197, 768]
            第 11 Layer Shape: [1, 197, 768]
            """
        
        return x


class Classifier_Head(nn.Layer):
    """分类头
        Params Info:
            embed_dims: 嵌入维度大小
            num_classes: 分类数 -- 输出大小
            mlp_ratio: MLP隐藏层伸缩比例 -- 预训练才有效
            dropout_rate: MLP神经元丢弃率 -- 预训练才有效
            act: 激活函数 -- 预训练才有效
            train_from_scratch: 使用预训练的分类头，否则使用微调的分类头
            add_softmax: 是否对输出添加softmax操作
        Forward Tips:
            - 针对输入的Patch数据，提取Classifier的Embeding数据
            - 将该数据用于线性映射，针对预训练或者微调执行不同的输出映射
            - 最后根据需要进行softmax
    """
    def __init__(self,
                 embed_dims=768,
                 num_classes=1000,
                 mlp_ratio=4,
                 dropout_rate=0.,
                 act=nn.GELU,
                 train_from_scratch=False,
                 add_softmax=False):
        super(Classifier_Head, self).__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.train_from_scratch = train_from_scratch
        self.add_softmax = add_softmax

        if train_from_scratch is True:
            self.head_layer = MLP(
                in_features=embed_dims,
                out_features=num_classes,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                act=act)
        else:
            self.head_layer = nn.Linear(in_features=embed_dims,
                                        out_features=num_classes)

        if add_softmax is True:
            self.softmax = nn.Softmax()

    def forward(self, inputs):
        # inputs: B, N, D
        x = inputs[:, 0]
        x = self.head_layer(x)

        if self.add_softmax is True:
            x = self.softmax(x)

        return x
        

class VIT(nn.Layer):
    """VIT模型结果复现
        Params Info:
            进入各子模块中查看即可
    """
    def __init__(self,
                 img_size=224,
                 num_classes=1000,
                 in_channels=3,
                 patch_size=16,
                 embed_dims=768,
                 num_layers=12,
                 mlp_ratio=4,
                 num_head=12,
                 attn_dropout_rate=0.,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 train_from_scratch=False,
                 add_softmax=False,
                 act=nn.GELU,
                 norm=nn.LayerNorm):
        super(VIT, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      in_channels=in_channels,
                                      patch_size=patch_size,
                                      embed_dims=embed_dims) # 1, 197, 768
        
        self.transfer_encoder = Encoder(num_layers=num_layers,
                                        embed_dims=embed_dims,
                                        mlp_ratio=mlp_ratio,
                                        num_head=num_head,
                                        attn_dropout_rate=attn_dropout_rate,
                                        dropout_rate=dropout_rate,
                                        droppath_rate=droppath_rate,
                                        act=act,
                                        norm=norm) # 1, 197, 768

        self.head = Classifier_Head(embed_dims=embed_dims,
                                    num_classes=num_classes,
                                    mlp_ratio=mlp_ratio,
                                    dropout_rate=dropout_rate,
                                    act=act,
                                    train_from_scratch=train_from_scratch,
                                    add_softmax=add_softmax) # 1, 1000

    def forward(self, inputs):
        x = self.patch_embed(inputs) # B, N, D
        x = self.transfer_encoder(x)
        x = self.head(x)
        return x


def VIT_Base(num_classes,
             img_size=224,
             in_channels=3,
             num_layers=12,
             patch_size=16,
             mlp_ratio=4,
             num_head=12,
             embed_dims=768):
    return VIT(img_size=img_size,
               num_classes=num_classes,
               in_channels=in_channels,
               num_layers=num_layers,
               num_head=num_head,
               patch_size=patch_size,
               mlp_ratio=mlp_ratio,
               embed_dims=embed_dims)


def VIT_Large(num_classes,
             img_size=224,
             in_channels=3,
             num_layers=24,
             patch_size=16,
             mlp_ratio=4,
             num_head=16,
             embed_dims=1024):
    return VIT(img_size=img_size,
               num_classes=num_classes,
               in_channels=in_channels,
               num_layers=num_layers,
               num_head=num_head,
               patch_size=patch_size,
               mlp_ratio=mlp_ratio,
               embed_dims=embed_dims)


def VIT_Huge(num_classes,
             img_size=224,
             in_channels=3,
             num_layers=32,
             patch_size=16,
             mlp_ratio=4,
             num_head=16,
             embed_dims=1280):
    return VIT(img_size=img_size,
               num_classes=num_classes,
               in_channels=in_channels,
               num_layers=num_layers,
               num_head=num_head,
               patch_size=patch_size,
               mlp_ratio=mlp_ratio,
               embed_dims=embed_dims)


if __name__ == "__main__":

    data = paddle.empty((1, 3, 224, 224))

    side_patch_num = 2
    patch_size = 224 // side_patch_num

    model = VIT_Large(num_classes=20, patch_size=patch_size)
    y_pred = model(data)
    print(y_pred.shape)
    """[1, 1000]"""
    """[1, 20]"""

