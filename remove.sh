#!/bin/bash
# 删除目标IP带宽限制工具 v1.0
# 用法: ./remove-bandwidth-limit.sh [网络接口] [目标IP]

if [ $# -ne 2 ]; then
  echo "错误: 参数不正确!"
  echo "用法: $0 <网络接口> <目标IP>"
  echo "示例: $0 eth0 192.168.1.100"
  exit 1
fi

DEV="$1"        # 网络接口 (如: eth0)
TARGET_IP="$2"  # 目标IP (如: 192.168.1.100)

# 检查root权限
if [ "$(id -u)" != "0" ]; then
  echo "错误: 此脚本需要root权限，请使用sudo运行!"
  exit 1
fi

echo "正在查找 $TARGET_IP 的带宽限制规则..."

# 找出目标IP关联的classid
CLASSID=$(tc filter show dev $DEV | grep -A1 "$TARGET_IP" | grep "flowid" | awk '{print $NF}')

if [ -z "$CLASSID" ]; then
  # 尝试替代匹配方式 (支持十六进制格式)
  HEX_IP=$(printf "%02x" ${TARGET_IP//./ })
  CLASSID=$(tc filter show dev $DEV | grep -A1 "$HEX_IP" | grep "flowid" | awk '{print $NF}')
  
  if [ -z "$CLASSID" ]; then
    echo "未找到 $TARGET_IP 的带宽限制规则"
    exit 2
  fi
fi

echo "找到带宽规则 (ClassID: $CLASSID)"

# 查找关联的filter preference
PREF=$(tc filter show dev $DEV | grep -B1 "$CLASSID" | grep "pref" | head -1 | awk '{print $5}' | tr -d ':')

if [ -z "$PREF" ]; then
  echo "警告: 未找到关联的过滤器，尝试直接删除类..."
fi

# 删除关联对象
echo "正在删除规则:"
echo "1. 删除过滤器"
sudo tc filter del dev $DEV pref $PREF parent 1: 2>/dev/null || true

echo "2. 删除流量类"
sudo tc class del dev $DEV parent 1: classid $CLASSID 2>/dev/null || true

echo "3. 删除队列"
sudo tc qdisc del dev $DEV parent $CLASSID 2>/dev/null || true

echo -e "\n$TARGET_IP 的带宽限制已成功移除!"
