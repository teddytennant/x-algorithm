package com.twitter.botmaker.compiler.thrift;

public final class TType {
  public static final byte STOP = org.apache.thrift.protocol.TType.STOP;
  public static final byte VOID = org.apache.thrift.protocol.TType.VOID;
  public static final byte BOOL = org.apache.thrift.protocol.TType.BOOL;
  public static final byte BYTE = org.apache.thrift.protocol.TType.BYTE;
  public static final byte DOUBLE = org.apache.thrift.protocol.TType.DOUBLE;
  public static final byte I16 = org.apache.thrift.protocol.TType.I16;
  public static final byte I32 = org.apache.thrift.protocol.TType.I32;
  public static final byte I64 = org.apache.thrift.protocol.TType.I64;
  public static final byte STRING = org.apache.thrift.protocol.TType.STRING;
  public static final byte STRUCT = org.apache.thrift.protocol.TType.STRUCT;
  public static final byte MAP = org.apache.thrift.protocol.TType.MAP;
  public static final byte SET = org.apache.thrift.protocol.TType.SET;
  public static final byte LIST = org.apache.thrift.protocol.TType.LIST;
  public static final byte ENUM = org.apache.thrift.protocol.TType.ENUM;

  public static final byte BINARY = 40;

  private TType() {
  }
}

