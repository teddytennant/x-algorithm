package com.twitter.botmaker.compiler;

import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.hash.Hasher;

import org.apache.commons.text.StringEscapeUtils;
import org.apache.commons.text.translate.UnicodeUnescaper;
import org.apache.thrift.TBase;
import org.apache.thrift.TEnum;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TCompactProtocol;
import org.apache.thrift.protocol.TField;
import org.apache.thrift.protocol.TList;
import org.apache.thrift.protocol.TMap;
import org.apache.thrift.protocol.TMessage;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.protocol.TProtocolFactory;
import org.apache.thrift.protocol.TSet;
import org.apache.thrift.protocol.TStruct;
import org.apache.thrift.transport.TTransport;

import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.thrift.FieldMetaData;
import com.twitter.botmaker.compiler.thrift.FieldValueMetaData;
import com.twitter.botmaker.compiler.tuples.Pair;
import com.twitter.botmaker.compiler.tuples.StructTuple;
import com.twitter.botmaker.compiler.tuples.Tuple;
import com.twitter.botmaker.compiler.types.Symbol;
import com.twitter.botmaker.compiler.types.ThriftType;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.scrooge.ThriftStruct;
import com.twitter.scrooge.ThriftStructCodec;

public abstract class Serializer<T> {

  private static final int MIN_BUF_CAPACITY = 65536;
  private static final int MAX_BUF_CAPACITY = 65536 * 256;
  private static final int BUF_CAPACITY = 65536 * 16;
  private static final ThreadLocal<ByteBuffer> BUF =
      ThreadLocal.withInitial(() -> ByteBuffer.allocate(MIN_BUF_CAPACITY));
  private static final UnicodeUnescaper UNICODE_UNESCAPER = new UnicodeUnescaper();

  private Serializer() {
  }

  public ByteBuffer serialize(T value, boolean nullable) throws Exception {
    return serialize(value, nullable, new TCompactProtocol.Factory());
  }

  public ByteBuffer serialize(T value,
                              boolean nullable,
                              TProtocolFactory tProtocolFactory) throws Exception {
    try {
      ByteBuffer buffer = BUF.get();
      buffer.clear();

      TByteBufferTransport transport = new TByteBufferTransport(buffer);
      TProtocol protocol = tProtocolFactory.getProtocol(transport);
      serialize(protocol, value, nullable);

      buffer.flip();
      ByteBuffer result = ByteBuffer.allocate(buffer.remaining());
      result.put(buffer);

      if (buffer.capacity() > BUF_CAPACITY) {
        buffer = ByteBuffer.allocate(BUF_CAPACITY);
        BUF.set(buffer);
      }
      result.flip();
      return result;
    } catch (BufferOverflowException be) {
      int cap = BUF.get().capacity() << 1;
      if (cap > MAX_BUF_CAPACITY) {
        throw be;
      }
      ByteBuffer buffer = ByteBuffer.allocate(cap);
      BUF.set(buffer);
      return serialize(value, nullable);
    }
  }

  public T deserialize(ByteBuffer buffer, boolean nullable) throws Exception {
    return deserialize(buffer, nullable, new TCompactProtocol.Factory());
  }

  public T deserialize(ByteBuffer buffer,
                       boolean nullable,
                       TProtocolFactory tProtocolFactory) throws Exception {
    TByteBufferTransport transport = new TByteBufferTransport(buffer);
    TProtocol protocol = tProtocolFactory.getProtocol(transport);
    return deserialize(protocol, nullable);
  }

  public Fingerprint computeFingerprint(T value) throws Exception {
    Hasher hasher = Fingerprint.newHasher();

    if (value != null) {
      hasher.putUnencodedChars(value.getClass().getName());
      computeFingerprint(hasher, value);
    } else {
      hasher.putLong(0L);
    }

    return new Fingerprint(0, hasher.hash().asBytes());
  }

  public abstract void computeFingerprint(Hasher hasher, T value) throws Exception;

  public String getScript(T value) throws Exception {

    GenScript script = new GenScript();
    genScript(script, value);

    if (script.imports.size() == 0) {
      return script.builder.toString();
    } else {
      StringBuffer sb = new StringBuffer();
      sb.append("{\n");
      for (String imp : script.imports) {
        sb.append("import ").append(imp).append(";\n");
      }
      sb.append("\n");
      sb.append(script.builder.toString());
      sb.append("\n}");
      return sb.toString();
    }
  }

  protected void genScript(GenScript script, T value) throws Exception {
    if (value == null) {
      script.builder.append("NULL");
    } else {
      toScript(script, value);
    }
  }

  protected void genStratoJson(GenScript script, T value) throws Exception {
    if (value == null) {
      script.builder.append("null");
    } else {
      toStratoJson(script, value);
    }
  }

  public String getStratoJson(T value) throws Exception {
    GenScript script = new GenScript();
    genStratoJson(script, value);
    return script.builder.toString();
  }

  protected T deserialize(TProtocol prot, boolean nullable) throws Exception {
    if (!nullable) {
      return deserialize(prot);
    }

    boolean isnull = prot.readBool();
    if (isnull) {
      return null;
    }

    return deserialize(prot);
  }

  protected void serialize(TProtocol prot, T value, boolean nullable) throws Exception {
    if (!nullable) {
      serialize(prot, value);
    } else if (value != null) {
      prot.writeBool(false);
      serialize(prot, value);
    } else {
      prot.writeBool(true);
    }
  }

  protected abstract void serialize(TProtocol prot, T value) throws Exception;

  protected abstract T deserialize(TProtocol prot) throws Exception;

  protected abstract void toScript(GenScript script, T value) throws Exception;

  protected abstract void toStratoJson(GenScript script, T value) throws Exception;

  public static final Serializer<String> STRING = new Serializer<String>() {

    @Override
    public void computeFingerprint(Hasher hasher, String value) {
      hasher.putUnencodedChars(value);
    }

    @Override
    protected void serialize(TProtocol prot, String value) throws TException {
      prot.writeString(value);
    }

    @Override
    protected String deserialize(TProtocol prot) throws TException {
      return prot.readString();
    }

    @Override
    protected void toScript(GenScript script, String value) {
      String escaped = UNICODE_UNESCAPER.translate(StringEscapeUtils.escapeJava(value));

      script.builder
          .append("\"")
          .append(escaped)
          .append("\"");
    }

    @Override
    protected void toStratoJson(GenScript script, String value) {
      String escaped = UNICODE_UNESCAPER.translate(StringEscapeUtils.escapeJava(value));
      script.builder.append("\"").append(escaped).append("\"");
    }
  };

  public static final Serializer<Symbol> SYMBOL = new Serializer<Symbol>() {
    @Override
    public void computeFingerprint(Hasher hasher, Symbol value) {
      hasher.putUnencodedChars(value.getValue());
    }

    @Override
    protected void serialize(TProtocol prot, Symbol value) throws TException {
      prot.writeString(value.getValue());
    }

    @Override
    protected Symbol deserialize(TProtocol prot) throws TException {
      return new Symbol(prot.readString());
    }

    @Override
    protected void toScript(GenScript script, Symbol value) {
      script.builder
          .append("`")
          .append(value.getValue())
          .append("`");
    }

    @Override
    protected void toStratoJson(GenScript script, Symbol value) {
      throw new IllegalArgumentException("Symbols are not supported in strato JSON");
    }
  };

  public static final Serializer<Long> LONG = new Serializer<Long>() {

    @Override
    public void computeFingerprint(Hasher hasher, Long value) {
      hasher.putLong(value);
    }

    @Override
    protected void serialize(TProtocol prot, Long value) throws TException {
      prot.writeI64(value);
    }

    @Override
    protected Long deserialize(TProtocol prot) throws TException {
      return prot.readI64();
    }

    @Override
    protected void toScript(GenScript script, Long value) {
      script.builder.append(value.toString());
    }

    @Override
    protected void toStratoJson(GenScript script, Long value) {
      script.builder.append("\"").append(value.toString()).append("\"");
    }
  };

  public static final Serializer<Integer> INTEGER = new Serializer<Integer>() {

    @Override
    public void computeFingerprint(Hasher hasher, Integer value) {
      hasher.putInt(value);
    }

    @Override
    protected void serialize(TProtocol prot, Integer value) throws TException {
      prot.writeI32(value);
    }

    @Override
    protected Integer deserialize(TProtocol prot) throws TException {
      return prot.readI32();
    }

    @Override
    protected void toScript(GenScript script, Integer value) {
      script.builder.append(value.toString());
    }

    @Override
    protected void toStratoJson(GenScript script, Integer value) {
      script.builder.append(value.toString());
    }
  };

  public static final Serializer<Short> SHORT = new Serializer<Short>() {

    @Override
    public void computeFingerprint(Hasher hasher, Short value) {
      hasher.putShort(value);
    }

    @Override
    protected void serialize(TProtocol prot, Short value) throws TException {
      prot.writeI16(value);
    }

    @Override
    protected Short deserialize(TProtocol prot) throws TException {
      return prot.readI16();
    }

    @Override
    protected void toScript(GenScript script, Short value) {
      script.builder.append(value.toString());
    }

    @Override
    protected void toStratoJson(GenScript script, Short value) {
      script.builder.append(value.toString());
    }
  };

  public static final Serializer<Byte> BYTE = new Serializer<Byte>() {

    @Override
    public void computeFingerprint(Hasher hasher, Byte value) {
      hasher.putByte(value);
    }

    @Override
    protected void serialize(TProtocol prot, Byte value) throws TException {
      prot.writeByte(value);
    }

    @Override
    protected Byte deserialize(TProtocol prot) throws TException {
      return prot.readByte();
    }

    @Override
    protected void toScript(GenScript script, Byte value) {
      script.builder.append(value.toString());
    }

    @Override
    protected void toStratoJson(GenScript script, Byte value) {
      script.builder.append(value.toString());
    }
  };

  public static final Serializer<Double> DOUBLE = new Serializer<Double>() {

    @Override
    public void computeFingerprint(Hasher hasher, Double value) {
      hasher.putDouble(value);
    }

    @Override
    protected void serialize(TProtocol prot, Double value) throws TException {
      prot.writeDouble(value);
    }

    @Override
    protected Double deserialize(TProtocol prot) throws TException {
      return prot.readDouble();
    }

    @Override
    protected void toScript(GenScript script, Double value) {
      script.builder.append(value.toString());
    }

    @Override
    protected void toStratoJson(GenScript script, Double value) {
      script.builder.append(value.toString());
    }
  };

  public static final Serializer<Boolean> BOOLEAN = new Serializer<Boolean>() {

    @Override
    public void computeFingerprint(Hasher hasher, Boolean value) {
      hasher.putBoolean(value);
    }

    @Override
    protected void serialize(TProtocol prot, Boolean value) throws TException {
      prot.writeBool(value);
    }

    @Override
    protected Boolean deserialize(TProtocol prot) throws TException {
      return prot.readBool();
    }

    @Override
    protected void toScript(GenScript script, Boolean value) {
      script.builder.append(value ? "TRUE" : "FALSE");
    }

    @Override
    protected void toStratoJson(GenScript script, Boolean value) {
      script.builder.append(value ? "true" : "false");
    }
  };

  public static final Serializer<ByteBuffer> BINARY = new Serializer<ByteBuffer>() {

    @Override
    public void computeFingerprint(Hasher hasher, ByteBuffer value) {
      value.mark();
      hasher.putInt(value.remaining());
      while (value.hasRemaining()) {
        hasher.putByte(value.get());
      }
      value.reset();
    }

    @Override
    protected void serialize(TProtocol prot, ByteBuffer value) throws TException {
      value.mark();
      prot.writeBinary(value);
      value.reset();
    }

    @Override
    protected ByteBuffer deserialize(TProtocol prot) throws TException {
      return prot.readBinary();
    }

    @Override
    protected void toScript(GenScript script, ByteBuffer value) {
      script.builder.append("NULL");
    }

    @Override
    protected void toStratoJson(GenScript script, ByteBuffer value) {
      script.builder
          .append("\"")
          .append(Base64.getEncoder().encodeToString(value.array()))
          .append("\"");
    }
  };

  public static final Serializer<Object> OBJECT = new Serializer<Object>() {

    private final byte tagBoolean = 1;
    private final byte tagString = 5;
    private final byte tagBinary = 7;
    private final byte tagFloat = 11;
    private final byte tagDouble = 12;
    private final byte tagLong = 20;
    private final byte tagInteger = 21;
    private final byte tagShort = 22;
    private final byte tagByte = 23;
    private final byte tagList = 30;
    private final byte tagSet = 31;
    private final byte tagMap = 32;
    private final byte tagPair = 33;
    private final byte tagTuple = 34;
    private final byte tagStructTuple = 35;
    private final byte tagTBase = 40;
    private final byte tagTEnum = 41;
    private final byte tagThriftStruct = 50;

    @Override
    public void computeFingerprint(Hasher hasher, Object value) throws Exception {
      if (value instanceof Boolean) {
        BOOLEAN.computeFingerprint(hasher, (Boolean) value);
      } else if (value instanceof String) {
        STRING.computeFingerprint(hasher, (String) value);
      } else if (value instanceof ByteBuffer) {
        BINARY.computeFingerprint(hasher, (ByteBuffer) value);
      } else if (value instanceof Float) {
        DOUBLE.computeFingerprint(hasher, ((Float) value).doubleValue());
      } else if (value instanceof Double) {
        DOUBLE.computeFingerprint(hasher, (Double) value);
      } else if (value instanceof Long) {
        LONG.computeFingerprint(hasher, (Long) value);
      } else if (value instanceof Integer) {
        INTEGER.computeFingerprint(hasher, (Integer) value);
      } else if (value instanceof Short) {
        SHORT.computeFingerprint(hasher, (Short) value);
      } else if (value instanceof Byte) {
        BYTE.computeFingerprint(hasher, (Byte) value);
      } else if (value instanceof List) {
        Serializer<List> listSerializer = listOf(Type.OBJECT);
        listSerializer.computeFingerprint(hasher, (List) value);
      } else if (value instanceof Set) {
        Serializer<Set> setSerializer = setOf(Type.OBJECT);
        setSerializer.computeFingerprint(hasher, (Set) value);
      } else if (value instanceof Map) {
        Serializer<Map> mapSerializer = mapOf(Type.OBJECT, Type.OBJECT);
        mapSerializer.computeFingerprint(hasher, (Map) value);
      } else if (value instanceof Pair) {
        Serializer<Pair> pairSerializer = pairOf(Type.OBJECT, Type.OBJECT);
        pairSerializer.computeFingerprint(hasher, (Pair) value);
      } else if (value instanceof StructTuple) {
        StructTuple structTuple = (StructTuple) value;
        Type[] typeParams = new Type[structTuple.size()];
        Arrays.fill(typeParams, Type.OBJECT);
        Serializer<StructTuple> tupleSerializer = structTupleOf(
            structTuple.getFieldNames(), ImmutableList.copyOf(typeParams));
        tupleSerializer.computeFingerprint(hasher, structTuple);
      } else if (value instanceof Tuple) {
        Tuple tuple = (Tuple) value;
        Type[] typeParams = new Type[tuple.size()];
        Arrays.fill(typeParams, Type.OBJECT);
        Serializer<Tuple> tupleSerializer = tupleOf(typeParams);
        tupleSerializer.computeFingerprint(hasher, (Tuple) value);
      } else if (value instanceof TBase) {
        Class<? extends TBase> clazz = (Class<? extends TBase>) value.getClass();
        Serializer serializer = thriftOf(clazz);
        serializer.computeFingerprint(hasher, value);
      } else if (value instanceof TEnum) {
        Class<? extends TEnum> clazz = (Class<? extends TEnum>) value.getClass();
        Serializer serializer = thriftEnumOf(clazz);
        serializer.computeFingerprint(hasher, value);
      } else if (value instanceof ThriftStruct) {
        Class<? extends ThriftStruct> clazz = (Class<? extends ThriftStruct>) value.getClass();
        Serializer serializer = thriftStructOf(clazz);
        serializer.computeFingerprint(hasher, value);
      } else {
        throw new UnsupportedOperationException(
            String.format("Unsupported type %s to fingerprint.", value.getClass().getName())
        );
      }
    }

    @Override
    protected void serialize(TProtocol prot, Object value) throws Exception {
      if (value instanceof Boolean) {
        prot.writeByte(tagBoolean);
        BOOLEAN.serialize(prot, (Boolean) value, false);
      } else if (value instanceof String) {
        prot.writeByte(tagString);
        STRING.serialize(prot, (String) value, false);
      } else if (value instanceof ByteBuffer) {
        prot.writeByte(tagBinary);
        BINARY.serialize(prot, (ByteBuffer) value, false);
      } else if (value instanceof Float) {
        prot.writeByte(tagFloat);
        DOUBLE.serialize(prot, ((Float) value).doubleValue(), false);
      } else if (value instanceof Double) {
        prot.writeByte(tagDouble);
        DOUBLE.serialize(prot, (Double) value, false);
      } else if (value instanceof Long) {
        prot.writeByte(tagLong);
        LONG.serialize(prot, (Long) value, false);
      } else if (value instanceof Integer) {
        prot.writeByte(tagInteger);
        INTEGER.serialize(prot, (Integer) value, false);
      } else if (value instanceof Short) {
        prot.writeByte(tagShort);
        SHORT.serialize(prot, (Short) value, false);
      } else if (value instanceof Byte) {
        prot.writeByte(tagByte);
        BYTE.serialize(prot, (Byte) value, false);
      } else if (value instanceof List) {
        prot.writeByte(tagList);
        Serializer<List> listSerializer = listOf(Type.OBJECT);
        listSerializer.serialize(prot, (List) value, false);
      } else if (value instanceof Set) {
        prot.writeByte(tagSet);
        Serializer<Set> setSerializer = setOf(Type.OBJECT);
        setSerializer.serialize(prot, (Set) value, false);
      } else if (value instanceof Map) {
        prot.writeByte(tagMap);
        Serializer<Map> mapSerializer = mapOf(Type.OBJECT, Type.OBJECT);
        mapSerializer.serialize(prot, (Map) value, false);
      } else if (value instanceof Pair) {
        prot.writeByte(tagPair);
        Serializer<Pair> pairSerializer = pairOf(Type.OBJECT, Type.OBJECT);
        pairSerializer.serialize(prot, (Pair) value, false);
      } else if (value instanceof StructTuple) {
        prot.writeByte(tagStructTuple);
        StructTuple structTuple = (StructTuple) value;
        Type[] typeParams = new Type[structTuple.size()];
        Arrays.fill(typeParams, Type.OBJECT);
        Serializer<StructTuple> structTupleSerializer =
            structTupleOf(structTuple.getFieldNames(), ImmutableList.copyOf(typeParams));
        prot.writeI32(structTuple.size());
        for (int i = 0; i < structTuple.size(); i++) {
          prot.writeString(structTuple.getFieldNames().get(i));
        }
        structTupleSerializer.serialize(prot, structTuple, false);
      } else if (value instanceof Tuple) {
        prot.writeByte(tagTuple);
        Tuple tuple = (Tuple) value;
        Type[] typeParams = new Type[tuple.size()];
        Arrays.fill(typeParams, Type.OBJECT);
        Serializer<Tuple> tupleSerializer = tupleOf(typeParams);
        prot.writeI32(tuple.size());
        tupleSerializer.serialize(prot, (Tuple) value, false);
      } else if (value instanceof TBase) {
        Class<? extends TBase> clazz = (Class<? extends TBase>) value.getClass();
        prot.writeByte(tagTBase);
        prot.writeString(clazz.getName());
        Serializer serializer = thriftOf(clazz);
        serializer.serialize(prot, value, false);
      } else if (value instanceof TEnum) {
        Class<? extends TEnum> clazz = (Class<? extends TEnum>) value.getClass();
        prot.writeByte(tagTEnum);
        prot.writeString(clazz.getName());
        Serializer serializer = thriftEnumOf(clazz);
        serializer.serialize(prot, value, false);
      } else if (value instanceof ThriftStruct) {
        Class<? extends ThriftStruct> clazz = (Class<? extends ThriftStruct>) value.getClass();
        prot.writeByte(tagThriftStruct);
        prot.writeString(clazz.getName());
        Serializer serializer = thriftStructOf(clazz);
        serializer.serialize(prot, value, false);
      } else {
        throw new UnsupportedOperationException(
            String.format("Unsupported type %s to serialize.", value.getClass().getName())
        );
      }
    }

    @Override
    protected Object deserialize(TProtocol prot) throws Exception {
      byte tag = prot.readByte();
      switch (tag) {
        case tagBoolean:
          return BOOLEAN.deserialize(prot);

        case tagString:
          return STRING.deserialize(prot);

        case tagBinary:
          return BINARY.deserialize(prot);

        case tagFloat:
          return DOUBLE.deserialize(prot).floatValue();

        case tagDouble:
          return DOUBLE.deserialize(prot);

        case tagLong:
          return LONG.deserialize(prot);

        case tagInteger:
          return INTEGER.deserialize(prot);

        case tagShort:
          return SHORT.deserialize(prot);

        case tagByte:
          return BYTE.deserialize(prot);

        case tagList:
          Serializer<List> listSerializer = listOf(Type.OBJECT);
          return listSerializer.deserialize(prot);

        case tagSet:
          Serializer<Set> setSerializer = setOf(Type.OBJECT);
          return setSerializer.deserialize(prot);

        case tagMap:
          Serializer<Map> mapSerializer = mapOf(Type.OBJECT, Type.OBJECT);
          return mapSerializer.deserialize(prot);

        case tagPair:
          Serializer<Pair> pairSerializer = pairOf(Type.OBJECT, Type.OBJECT);
          return pairSerializer.deserialize(prot);

        case tagStructTuple:
          int structTupleSize = prot.readI32();
          List<String> fieldNames = Lists.newArrayListWithCapacity(structTupleSize);
          for (int i = 0; i < structTupleSize; i++) {
            fieldNames.add(prot.readString());
          }

          Type[] structTupleTypeParams = new Type[structTupleSize];
          Arrays.fill(structTupleTypeParams, Type.OBJECT);
          Serializer<StructTuple> structTupleSerializer = structTupleOf(
              ImmutableList.copyOf(fieldNames),
              ImmutableList.copyOf(structTupleTypeParams));
          return structTupleSerializer.deserialize(prot);

        case tagTuple:
          int tupleSize = prot.readI32();
          Type[] typeParams = new Type[tupleSize];
          Arrays.fill(typeParams, Type.OBJECT);
          Serializer<Tuple> tupleSerializer = tupleOf(typeParams);
          return tupleSerializer.deserialize(prot);

        case tagTBase:
          String tbaseClassName = prot.readString();
          Class<?> tbaseClazz = ClassCache.forName(tbaseClassName);
          Serializer tbaseSerializer =
              thriftOf((Class<? extends TBase>) tbaseClazz);
          return tbaseSerializer.deserialize(prot, false);

        case tagTEnum:
          String tenumClassName = prot.readString();
          Class<?> tenumClazz = ClassCache.forName(tenumClassName);
          Serializer tenumSerializer =
              thriftEnumOf((Class<? extends TEnum>) tenumClazz);
          return tenumSerializer.deserialize(prot, false);

        case tagThriftStruct:
          String tstructClassName = prot.readString();
          Class<?> tstructClazz = ClassCache.forName(tstructClassName);
          Serializer tstructSerializer =
              thriftStructOf((Class<? extends ThriftStruct>) tstructClazz);
          return tstructSerializer.deserialize(prot, false);

        default:
          throw new UnsupportedOperationException(
              String.format("Unknown tag %d", tag)
          );
      }
    }

    @Override
    protected void toScript(GenScript script, Object value) throws Exception {
      if (value instanceof Boolean) {
        BOOLEAN.genScript(script, (Boolean) value);
      } else if (value instanceof String) {
        STRING.genScript(script, (String) value);
      } else if (value instanceof ByteBuffer) {
        BINARY.genScript(script, (ByteBuffer) value);
      } else if (value instanceof Float) {
        DOUBLE.genScript(script, ((Float) value).doubleValue());
      } else if (value instanceof Double) {
        DOUBLE.genScript(script, (Double) value);
      } else if (value instanceof Long) {
        LONG.genScript(script, (Long) value);
      } else if (value instanceof Integer) {
        INTEGER.genScript(script, (Integer) value);
      } else if (value instanceof Short) {
        SHORT.genScript(script, (Short) value);
      } else if (value instanceof Byte) {
        BYTE.genScript(script, (Byte) value);
      } else if (value instanceof List) {
        Serializer<List> listSerializer = listOf(Type.OBJECT);
        listSerializer.genScript(script, (List) value);
      } else if (value instanceof Set) {
        Serializer<Set> setSerializer = setOf(Type.OBJECT);
        setSerializer.genScript(script, (Set) value);
      } else if (value instanceof Map) {
        Serializer<Map> mapSerializer = mapOf(Type.OBJECT, Type.OBJECT);
        mapSerializer.genScript(script, (Map) value);
      } else if (value instanceof Pair) {
        Serializer<Pair> pairSerializer = pairOf(Type.OBJECT, Type.OBJECT);
        pairSerializer.genScript(script, (Pair) value);
      } else if (value instanceof StructTuple) {
        StructTuple structTuple = (StructTuple) value;
        Type[] typeParams = new Type[structTuple.size()];
        Arrays.fill(typeParams, Type.OBJECT);
        Serializer<StructTuple> tupleSerializer = structTupleOf(
            structTuple.getFieldNames(), ImmutableList.copyOf(typeParams));
        tupleSerializer.genScript(script, structTuple);
      } else if (value instanceof Tuple) {
        Tuple tuple = (Tuple) value;
        Type[] typeParams = new Type[tuple.size()];
        Arrays.fill(typeParams, Type.OBJECT);
        Serializer<Tuple> tupleSerializer = tupleOf(typeParams);
        tupleSerializer.genScript(script, (Tuple) value);
      } else if (value instanceof TBase) {
        Class<? extends TBase> clazz = (Class<? extends TBase>) value.getClass();
        Serializer serializer = thriftOf(clazz);
        serializer.genScript(script, value);
      } else if (value instanceof TEnum) {
        Class<? extends TEnum> clazz = (Class<? extends TEnum>) value.getClass();
        Serializer serializer = thriftEnumOf(clazz);
        serializer.genScript(script, value);
      } else if (value instanceof ThriftStruct) {
        Class<? extends ThriftStruct> clazz = (Class<? extends ThriftStruct>) value.getClass();
        Serializer serializer = thriftStructOf(clazz);
        serializer.genScript(script, value);
      } else {
        throw new UnsupportedOperationException(
            String.format("Unsupported type %s to fingerprint.", value.getClass().getName())
        );
      }
    }

    @Override
    protected void toStratoJson(GenScript script, Object value) throws Exception {
      if (value instanceof Boolean) {
        BOOLEAN.genStratoJson(script, (Boolean) value);
      } else if (value instanceof String) {
        STRING.genStratoJson(script, (String) value);
      } else if (value instanceof ByteBuffer) {
        BINARY.genStratoJson(script, (ByteBuffer) value);
      } else if (value instanceof Float) {
        DOUBLE.genStratoJson(script, ((Float) value).doubleValue());
      } else if (value instanceof Double) {
        DOUBLE.genStratoJson(script, (Double) value);
      } else if (value instanceof Long) {
        LONG.genStratoJson(script, (Long) value);
      } else if (value instanceof Integer) {
        INTEGER.genStratoJson(script, (Integer) value);
      } else if (value instanceof Short) {
        SHORT.genStratoJson(script, (Short) value);
      } else if (value instanceof Byte) {
        BYTE.genStratoJson(script, (Byte) value);
      } else if (value instanceof List) {
        Serializer<List> listSerializer = listOf(Type.OBJECT);
        listSerializer.genStratoJson(script, (List) value);
      } else if (value instanceof Set) {
        Serializer<Set> setSerializer = setOf(Type.OBJECT);
        setSerializer.genStratoJson(script, (Set) value);
      } else if (value instanceof Map) {
        Serializer<Map> mapSerializer = mapOf(Type.OBJECT, Type.OBJECT);
        mapSerializer.genStratoJson(script, (Map) value);
      } else if (value instanceof Pair) {
        Serializer<Pair> pairSerializer = pairOf(Type.OBJECT, Type.OBJECT);
        pairSerializer.genStratoJson(script, (Pair) value);
      } else if (value instanceof StructTuple) {
        StructTuple structTuple = (StructTuple) value;
        Type[] typeParams = new Type[structTuple.size()];
        Arrays.fill(typeParams, Type.OBJECT);
        Serializer<StructTuple> tupleSerializer = structTupleOf(
            structTuple.getFieldNames(), ImmutableList.copyOf(typeParams));
        tupleSerializer.genStratoJson(script, structTuple);
      } else if (value instanceof Tuple) {
        Tuple tuple = (Tuple) value;
        Type[] typeParams = new Type[tuple.size()];
        Arrays.fill(typeParams, Type.OBJECT);
        Serializer<Tuple> tupleSerializer = tupleOf(typeParams);
        tupleSerializer.genStratoJson(script, (Tuple) value);
      } else if (value instanceof TBase) {
        Class<? extends TBase> clazz = (Class<? extends TBase>) value.getClass();
        Serializer serializer = thriftOf(clazz);
        serializer.genStratoJson(script, value);
      } else if (value instanceof TEnum) {
        Class<? extends TEnum> clazz = (Class<? extends TEnum>) value.getClass();
        Serializer serializer = thriftEnumOf(clazz);
        serializer.genStratoJson(script, value);
      } else if (value instanceof ThriftStruct) {
        Class<? extends ThriftStruct> clazz = (Class<? extends ThriftStruct>) value.getClass();
        Serializer serializer = thriftStructOf(clazz);
        serializer.genStratoJson(script, value);
      } else {
        throw new UnsupportedOperationException(
            String.format("Unsupported type %s to fingerprint.", value.getClass().getName())
        );
      }
    }
  };

  public static Serializer<Collection> collectionOf(Type type) {
    return new Serializer<Collection>() {

      @Override
      public void computeFingerprint(Hasher hasher, Collection value) throws Exception {
        type.computerFingerprint(hasher);
        hasher.putInt(value.size());
        int index = 0;
        for (Object item : value) {
          if (item != null) {
            type.serializer.computeFingerprint(hasher, item);
          }
          hasher.putInt(index++);
        }
      }

      @Override
      protected void serialize(TProtocol prot, Collection value) throws Exception {

        boolean nullable = false;
        for (Object item : value) {
          if (item == null) {
            nullable = true;
            break;
          }
        }
        prot.writeBool(nullable);
        prot.writeI32(value.size());
        for (Object item : value) {
          type.serializer.serialize(prot, item, nullable);
        }
      }

      @Override
      protected Collection deserialize(TProtocol prot) throws Exception {

        boolean nullable = prot.readBool();
        int size = prot.readI32();
        List<Object> list = Lists.newArrayListWithCapacity(size);
        for (int i = 0; i < size; i++) {
          list.add(type.serializer.deserialize(prot, nullable));
        }

        return list;
      }

      @Override
      protected void toScript(GenScript script, Collection value) throws Exception {
        script.builder.append("[");
        script.level++;

        int index = 0;
        for (Object item : value) {
          if (index++ > 0) {
            script.builder.append(", ");
          }
          type.serializer.genScript(script, item);
        }

        script.level--;
        script.builder.append("]");
      }

      @Override
      protected void toStratoJson(GenScript script, Collection collection) throws Exception {
        script.builder.append("[ ");
        boolean first = true;
        for (Object item : collection) {
          if (first) {
            first = false;
          } else {
            script.builder.append(", ");
          }
          type.serializer.genStratoJson(script, item);
        }
        script.builder.append(" ]");
      }
    };
  }

  public static Serializer<List> listOf(Type type) {
    return new Serializer<List>() {

      @Override
      public void computeFingerprint(Hasher hasher, List value) throws Exception {
        type.computerFingerprint(hasher);
        hasher.putInt(value.size());
        int index = 0;
        for (Object item : value) {
          if (item != null) {
            type.serializer.computeFingerprint(hasher, item);
          }
          hasher.putInt(index++);
        }
      }

      @Override
      protected void serialize(TProtocol prot, List value) throws Exception {

        boolean nullable = false;
        for (Object item : value) {
          if (item == null) {
            nullable = true;
            break;
          }
        }

        prot.writeBool(nullable);
        prot.writeI32(value.size());
        for (Object item : value) {
          type.serializer.serialize(prot, item, nullable);
        }
      }

      @Override
      protected List deserialize(TProtocol prot) throws Exception {

        boolean nullable = prot.readBool();
        int size = prot.readI32();
        List<Object> list = Lists.newArrayListWithCapacity(size);
        for (int i = 0; i < size; i++) {
          list.add(type.serializer.deserialize(prot, nullable));
        }

        return list;
      }

      @Override
      protected void toScript(GenScript script, List list) throws Exception {
        script.builder.append("[");
        script.level++;

        int index = 0;
        for (Object item : list) {
          if (index++ > 0) {
            script.builder.append(", ");
          }
          type.serializer.genScript(script, item);
        }

        script.level--;
        script.builder.append("]");
      }

      @Override
      protected void toStratoJson(GenScript script, List list) throws Exception {
        script.builder.append("[ ");
        boolean first = true;
        for (Object item : list) {
          if (first) {
            first = false;
          } else {
            script.builder.append(", ");
          }
          type.serializer.genStratoJson(script, item);
        }
        script.builder.append(" ]");
      }
    };
  }

  public static Serializer<Set> setOf(Type type) {
    return new Serializer<Set>() {

      @Override
      public void computeFingerprint(Hasher hasher, Set set) throws Exception {
        type.computerFingerprint(hasher);
        hasher.putInt(set.size());
        int index = 0;
        for (Object item : set) {
          if (item != null) {
            type.serializer.computeFingerprint(hasher, item);
          }
          hasher.putInt(index++);
        }
      }

      @Override
      protected void serialize(TProtocol prot, Set set) throws Exception {

        boolean nullable = false;
        for (Object item : set) {
          if (item == null) {
            nullable = true;
            break;
          }
        }

        prot.writeBool(nullable);
        prot.writeI32(set.size());
        for (Object item : set) {
          type.serializer.serialize(prot, item, nullable);
        }
      }

      @Override
      protected Set deserialize(TProtocol prot) throws Exception {

        boolean nullable = prot.readBool();
        int size = prot.readI32();
        Set<Object> set = Sets.newHashSetWithExpectedSize(size);
        for (int i = 0; i < size; i++) {
          set.add(type.serializer.deserialize(prot, nullable));
        }

        return set;
      }

      @Override
      protected void toScript(GenScript script, Set set) throws Exception {

        script.builder.append("Set(");
        script.level++;

        int index = 0;
        for (Object item : set) {
          if (index++ > 0) {
            script.builder.append(", ");
          }
          type.serializer.genScript(script, item);
        }

        script.level--;
        script.builder.append(")");
      }

      @Override
      protected void toStratoJson(GenScript script, Set set) throws Exception {
        script.builder.append("[ ");
        boolean first = true;
        for (Object item : set) {
          if (first) {
            first = false;
          } else {
            script.builder.append(", ");
          }
          type.serializer.genStratoJson(script, item);
        }
        script.builder.append(" ]");
      }
    };
  }

  public static Serializer<Map> mapOf(Type keyType, Type valType) {
    return new Serializer<Map>() {

      @Override
      public void computeFingerprint(Hasher hasher, Map map) throws Exception {
        keyType.computerFingerprint(hasher);
        valType.computerFingerprint(hasher);
        hasher.putInt(map.size());
        int index = 0;
        for (Object key : map.keySet()) {
          Object value = map.get(key);
          hasher.putInt(index++);
          if (key != null) {
            keyType.serializer.computeFingerprint(hasher, key);
          }
          hasher.putInt(index++);
          if (value != null) {
            valType.serializer.computeFingerprint(hasher, value);
          }
        }
      }

      @Override
      protected void serialize(TProtocol prot, Map map) throws Exception {

        boolean keyNullable = false;
        for (Object key : map.keySet()) {
          if (key == null) {
            keyNullable = true;
            break;
          }
        }

        boolean valueNullable = false;
        for (Object value : map.values()) {
          if (value == null) {
            valueNullable = true;
            break;
          }
        }

        prot.writeBool(keyNullable);
        prot.writeBool(valueNullable);
        prot.writeI32(map.size());
        for (Object key : map.keySet()) {
          keyType.serializer.serialize(prot, key, keyNullable);
          valType.serializer.serialize(prot, map.get(key), valueNullable);
        }
      }

      @Override
      protected Map deserialize(TProtocol prot) throws Exception {

        boolean keyNullable = prot.readBool();
        boolean valueNullable = prot.readBool();
        int size = prot.readI32();
        Map<Object, Object> map = Maps.newHashMapWithExpectedSize(size);
        for (int i = 0; i < size; i++) {
          map.put(
              keyType.serializer.deserialize(prot, keyNullable),
              valType.serializer.deserialize(prot, valueNullable));
        }

        return map;
      }

      @Override
      protected void toScript(GenScript script, Map map) throws Exception {

        if (script.level == 0 && map.size() > 1) {

          script.builder.append("Map(");
          script.builder.append("\n");
          script.level++;

          int index = 0;
          for (Object key : map.keySet()) {

            if (index++ > 0) {
              script.builder.append(",");
              script.builder.append("\n");
            }

            script.indent();

            Object value = map.get(key);
            keyType.serializer.genScript(script, key);
            script.builder.append(": ");
            valType.serializer.genScript(script, value);
          }

          script.level--;
          script.builder.append("\n");
          script.builder.append(")");

        } else {
          script.builder.append("Map(");
          script.level++;

          int index = 0;
          for (Object key : map.keySet()) {

            if (index++ > 0) {
              script.builder.append(", ");
            }

            Object value = map.get(key);
            keyType.serializer.genScript(script, key);
            script.builder.append(": ");
            valType.serializer.genScript(script, value);
          }

          script.level--;
          script.builder.append(")");
        }
      }

      @Override
      protected void toStratoJson(GenScript script, Map map) throws Exception {
        if (map.isEmpty()) {
          script.builder.append("{}");
        }
        else if (map.keySet().stream().allMatch(element -> element instanceof String)) {
          script.builder.append("{ ");
          boolean first = true;
          for (Object key : map.keySet()) {
            if (first) {
              first = false;
            } else {
              script.builder.append(", ");
            }
            keyType.serializer.genStratoJson(script, key);
            script.builder.append(": ");
            Object value = map.get(key);
            valType.serializer.genStratoJson(script, value);
          }
          script.builder.append(" }");
        } else {
          script.builder.append("[ ");
          boolean first = true;
          for (Object key : map.keySet()) {
            if (first) {
              first = false;
            } else {
              script.builder.append(", ");
            }
            script.builder.append("[");
            keyType.serializer.genStratoJson(script, key);
            script.builder.append(", ");
            Object value = map.get(key);
            valType.serializer.genStratoJson(script, value);
            script.builder.append("]");
          }
          script.builder.append(" ]");
        }
      }
    };
  }

  public static Serializer<Pair> pairOf(Type keyType, Type valType) {
    return new Serializer<Pair>() {

      @Override
      public void computeFingerprint(Hasher hasher, Pair pair) throws Exception {
        keyType.computerFingerprint(hasher);
        valType.computerFingerprint(hasher);

        int index = 0;
        hasher.putInt(index++);
        if (pair.getFirst() != null) {
          keyType.serializer.computeFingerprint(hasher, pair.getFirst());
        }
        hasher.putInt(index++);
        if (pair.getSecond() != null) {
          valType.serializer.computeFingerprint(hasher, pair.getSecond());
        }
      }

      @Override
      protected void serialize(TProtocol prot, Pair pair) throws Exception {

        keyType.serializer.serialize(prot, pair.getFirst(), true);
        valType.serializer.serialize(prot, pair.getSecond(), true);
      }

      @Override
      protected Pair deserialize(TProtocol prot) throws Exception {

        return Pair.of(
            keyType.serializer.deserialize(prot, true),
            valType.serializer.deserialize(prot, true));
      }

      @Override
      protected void toScript(GenScript script, Pair pair) throws Exception {
        keyType.serializer.genScript(script, pair.getFirst());
        script.builder.append(": ");
        valType.serializer.genScript(script, pair.getSecond());
      }

      @Override
      protected void toStratoJson(GenScript script, Pair pair) {
        throw new IllegalArgumentException("Pairs are not supported in strato JSON");
      }
    };
  }

  public static Serializer<Tuple> tupleOf(Type... typeParams) {
    return new Serializer<Tuple>() {

      @Override
      public void computeFingerprint(Hasher hasher, Tuple tuple) throws Exception {
        for (Type type : typeParams) {
          type.computerFingerprint(hasher);
        }

        for (int index = 0; index < tuple.size(); index++) {
          hasher.putInt(index);
          if (tuple.get(index) != null) {
            typeParams[index].serializer.computeFingerprint(hasher, tuple.get(index));
          }
        }
      }

      @Override
      protected void serialize(TProtocol prot, Tuple tuple) throws Exception {

        for (int index = 0; index < typeParams.length; index++) {
          typeParams[index].serializer.serialize(prot, tuple.get(index), true);
        }
      }

      @Override
      protected Tuple deserialize(TProtocol prot) throws Exception {

        List<Object> values = Lists.newArrayListWithCapacity(typeParams.length);
        for (int index = 0; index < typeParams.length; index++) {
          Object value = typeParams[index].serializer.deserialize(prot, true);
          values.add(value);
        }

        return Tuple.of(values.toArray());
      }

      @Override
      protected void toScript(GenScript script, Tuple tuple) throws Exception {
        script.builder.append("Tuple(");
        script.level++;

        for (int index = 0; index < tuple.size(); index++) {
          if (index > 0) {
            script.builder.append(", ");
          }
          Object item = tuple.get(index);
          typeParams[index].serializer.genScript(script, item);
        }

        script.level--;
        script.builder.append(")");
      }

      @Override
      protected void toStratoJson(GenScript script, Tuple tuple) {
        throw new IllegalArgumentException("Tuples are not supported in strato JSON");
      }
    };
  }

  public static Serializer<StructTuple> structTupleOf(
      ImmutableList<String> fieldNames, ImmutableList<Type> fieldTypes) {
    Preconditions.checkArgument(fieldNames.size() == fieldTypes.size());
    return new Serializer<StructTuple>() {

      @Override
      public void computeFingerprint(Hasher hasher, StructTuple structTuple) throws Exception {

        for (int index = 0; index < structTuple.size(); index++) {
          hasher.putInt(index);
          hasher.putUnencodedChars(fieldNames.get(index));
          fieldTypes.get(index).computerFingerprint(hasher);
          if (structTuple.get(index) != null) {
            fieldTypes.get(index).serializer.computeFingerprint(hasher, structTuple.get(index));
          }
        }
      }

      @Override
      protected void serialize(TProtocol prot, StructTuple structTuple) throws Exception {

        for (int index = 0; index < fieldTypes.size(); index++) {
          fieldTypes.get(index).serializer.serialize(prot, structTuple.get(index), true);
        }
      }

      @Override
      protected StructTuple deserialize(TProtocol prot) throws Exception {

        List<Object> values = Lists.newArrayListWithCapacity(fieldNames.size());
        for (int index = 0; index < fieldTypes.size(); index++) {
          Object value = fieldTypes.get(index).serializer.deserialize(prot, true);
          values.add(value);
        }

        return StructTuple.of(fieldNames, values);
      }

      @Override
      protected void toScript(GenScript script, StructTuple structTuple) throws Exception {
        script.builder.append("StructTuple(");
        script.level++;

        for (int index = 0; index < structTuple.size(); index++) {
          if (index > 0) {
            script.builder.append(", ");
          }
          script.builder.append('"');
          script.builder.append(fieldNames.get(index));
          script.builder.append('"');
          script.builder.append(": ");
          Object item = structTuple.get(index);
          fieldTypes.get(index).serializer.genScript(script, item);
        }

        script.level--;
        script.builder.append(")");
      }

      @Override
      protected void toStratoJson(GenScript script, StructTuple structTuple) {
        throw new IllegalArgumentException("StructTuples are not supported in strato JSON");
      }
    };
  }

  public static <T extends TBase> Serializer<T> thriftOf(
      Class<T> clazz) throws SemanticCheckFailure {

    Map<String, FieldMetaData> metaDataMap = ThriftType.getMetaData(clazz);

    return new Serializer<T>() {

      @Override
      public void computeFingerprint(Hasher hasher, T value) throws Exception {
        TFingerprintProtocol prot = new TFingerprintProtocol(hasher);
        value.write(prot);
      }

      @Override
      protected void serialize(TProtocol prot, TBase value) throws TException {
        value.write(prot);
      }

      @Override
      protected T deserialize(TProtocol prot) throws Exception {
        T value = clazz.newInstance();
        value.read(prot);
        return value;
      }

      @Override
      protected void toScript(GenScript script, T thriftObj) throws Exception {
        if (script.level == 0) {
          script.imports.add(clazz.getName());
          script.builder.append(clazz.getSimpleName()).append("(");
          script.builder.append("\n");
          script.level++;

          int index = 0;
          for (String fieldName : metaDataMap.keySet()) {

            FieldMetaData metaData = metaDataMap.get(fieldName);
            if (!thriftObj.isSet(thriftObj.fieldForId(metaData.fieldId))) {
              continue;
            }

            if (index++ > 0) {
              script.builder.append(",");
              script.builder.append("\n");
            }

            script.indent();

            FieldValueMetaData valueMetaData = metaData.valueMetaData;
            Type fieldType = valueMetaData.getRawType();
            Object fieldValue = thriftObj.getFieldValue(thriftObj.fieldForId(metaData.fieldId));
            Type.STRING.serializer.genScript(script, fieldName);
            script.builder.append(": ");

            if (fieldType == Type.STRING && fieldValue != null) {
              fieldType.serializer.genScript(script, fieldValue.toString());
            } else {
              fieldType.serializer.genScript(script, fieldValue);
            }
          }

          script.level--;
          script.builder.append("\n");
          script.builder.append(")");

        } else {
          script.imports.add(clazz.getName());
          script.builder.append(clazz.getSimpleName()).append("(");
          script.level++;

          int index = 0;
          for (String fieldName : metaDataMap.keySet()) {

            FieldMetaData metaData = metaDataMap.get(fieldName);
            if (!thriftObj.isSet(thriftObj.fieldForId(metaData.fieldId))) {
              continue;
            }

            if (index++ > 0) {
              script.builder.append(", ");
            }

            FieldValueMetaData valueMetaData = metaData.valueMetaData;
            Type fieldType = valueMetaData.getRawType();
            Object fieldValue = thriftObj.getFieldValue(thriftObj.fieldForId(metaData.fieldId));
            Type.STRING.serializer.genScript(script, fieldName);
            script.builder.append(": ");

            if (fieldType == Type.STRING && fieldValue != null) {
              fieldType.serializer.genScript(script, fieldValue.toString());
            } else {
              fieldType.serializer.genScript(script, fieldValue);
            }
          }

          script.level--;
          script.builder.append(")");
        }
      }

      @Override
      protected void toStratoJson(GenScript script, T thriftjava) {
        throw new IllegalArgumentException("Thriftjava objects are not supported in strato JSON");
      }
    };
  }

  public static <T extends TEnum> Serializer<T> thriftEnumOf(Class<T> typeEnum) {
    ImmutableMap.Builder<Integer, T> builder = ImmutableMap.builder();
    for (Object v : EnumSet.allOf((Class<? extends Enum>) typeEnum)) {
      T e = (T) v;
      builder.put(e.getValue(), e);
    }

    Map<Integer, T> enumMap = builder.build();

    return new Serializer<T>() {

      @Override
      public void computeFingerprint(Hasher hasher, T value) {
        hasher.putInt(value.getValue());
      }

      @Override
      protected void serialize(TProtocol prot, T value) throws Exception {
        prot.writeI32(value.getValue());
      }

      @Override
      protected T deserialize(TProtocol prot) throws Exception {
        int v = prot.readI32();
        return enumMap.get(v);
      }

      @Override
      protected void toScript(GenScript script, T value) {
        script.builder.append("NULL");
      }

      @Override
      protected void toStratoJson(GenScript script, T value) {
        throw new IllegalArgumentException("Thrift enums are not supported in strato JSON");
      }
    };
  }

  public static <T extends ThriftStruct> Serializer<T> thriftStructOf(Class<T> clazz)
      throws SemanticCheckFailure {
    try {
      ThriftStructCodec codec = ThriftStructCodec.forStructClass(clazz);

      return new Serializer<T>() {

        @Override
        public void computeFingerprint(Hasher hasher, T value) throws TException {
          TFingerprintProtocol prot = new TFingerprintProtocol(hasher);
          codec.encode(value, prot);
        }

        @Override
        protected void serialize(TProtocol prot, ThriftStruct value) throws TException {
          codec.encode(value, prot);
        }

        @Override
        protected T deserialize(TProtocol prot) throws TException {
          return (T) codec.decode(prot);
        }

        @Override
        protected void toScript(GenScript script, T value) {
          script.builder.append("NULL");
        }

        @Override
        protected void toStratoJson(GenScript script, T value) {
          throw new IllegalArgumentException(
              "Thriftscala objects are not supported in strato JSON");
        }
      };
    } catch (Exception e) {
      throw new SemanticCheckFailure(
          String.format("Failed to create serializer for ThriftStruct %s: %s",
              clazz.getName(), e.toString())
      );
    }
  }

  private static final class TByteBufferTransport extends TTransport {
    private ByteBuffer buffer;

    public TByteBufferTransport(ByteBuffer buf) {
      this.buffer = buf;
    }

    @Override
    public void close() {
    }

    @Override
    public boolean isOpen() {
      return true;
    }

    @Override
    public void open() {
    }

    @Override
    public int read(byte[] buf, int off, int len) {
      int read = buffer.remaining() > len ? len : buffer.remaining();
      buffer.get(buf, off, read);
      return read;
    }

    @Override
    public void write(byte[] buf, int off, int len) {
      buffer.put(buf, off, len);
    }
  }

  private static final class TFingerprintProtocol extends TProtocol {
    private final Hasher hasher;

    public TFingerprintProtocol(Hasher hasher) {
      super(null);
      this.hasher = hasher;
    }

    @Override
    public void writeMessageBegin(TMessage tMessage) {
      hasher.putInt(1);
      hasher.putByte(tMessage.type);
    }

    @Override
    public void writeMessageEnd() {
      hasher.putInt(-1);
    }

    @Override
    public void writeStructBegin(TStruct tStruct) {
      hasher.putInt(2);
    }

    @Override
    public void writeStructEnd() {
      hasher.putInt(-2);
    }

    @Override
    public void writeFieldBegin(TField tField) {
      hasher.putInt(3);
      hasher.putShort(tField.id);
      hasher.putByte(tField.type);
    }

    @Override
    public void writeFieldEnd() {
      hasher.putInt(-3);
    }

    @Override
    public void writeFieldStop() {
      hasher.putInt(-4);
    }

    @Override
    public void writeMapBegin(TMap tMap) {
      hasher.putByte(tMap.keyType);
      hasher.putByte(tMap.valueType);
      hasher.putInt(tMap.size);
    }

    @Override
    public void writeMapEnd() {
      hasher.putInt(-5);
    }

    @Override
    public void writeListBegin(TList tList) {
      hasher.putByte(tList.elemType);
      hasher.putInt(tList.size);
    }

    @Override
    public void writeListEnd() {
      hasher.putInt(-6);
    }

    @Override
    public void writeSetBegin(TSet tSet) {
      hasher.putByte(tSet.elemType);
      hasher.putInt(tSet.size);
    }

    @Override
    public void writeSetEnd() {
      hasher.putInt(-7);
    }

    @Override
    public void writeBool(boolean b) {
      hasher.putBoolean(b);
    }

    @Override
    public void writeByte(byte b) {
      hasher.putByte(b);
    }

    @Override
    public void writeI16(short i) {
      hasher.putShort(i);
    }

    @Override
    public void writeI32(int i) {
      hasher.putInt(i);
    }

    @Override
    public void writeI64(long l) {
      hasher.putLong(l);
    }

    @Override
    public void writeDouble(double v) {
      hasher.putDouble(v);
    }

    @Override
    public void writeString(String s) {
      hasher.putUnencodedChars(s);
    }

    @Override
    public void writeBinary(ByteBuffer byteBuffer) {
      byteBuffer.mark();
      hasher.putInt(byteBuffer.remaining());
      while (byteBuffer.hasRemaining()) {
        hasher.putByte(byteBuffer.get());
      }
      byteBuffer.reset();
    }

    @Override
    public TMessage readMessageBegin() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public void readMessageEnd() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public TStruct readStructBegin() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public void readStructEnd() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public TField readFieldBegin() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public void readFieldEnd() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public TMap readMapBegin() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public void readMapEnd() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public TList readListBegin() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public void readListEnd() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public TSet readSetBegin() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public void readSetEnd() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public boolean readBool() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public byte readByte() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public short readI16() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public int readI32() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public long readI64() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public double readDouble() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public String readString() throws TException {
      throw new TException(new UnsupportedOperationException());
    }

    @Override
    public ByteBuffer readBinary() throws TException {
      throw new TException(new UnsupportedOperationException());
    }
  }

  static class GenScript {

    public String indentStr = "  ";
    public StringBuffer builder = new StringBuffer();
    public Set<String> imports = Sets.newHashSet();
    public int level = 0;

    public void indent() {
      builder.append(indentStr);
    }
  }
}
