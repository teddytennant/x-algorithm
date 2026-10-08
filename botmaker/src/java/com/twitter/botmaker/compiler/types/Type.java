package com.twitter.botmaker.compiler.types;

import java.nio.ByteBuffer;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

import scala.runtime.BoxedUnit;
import scala.runtime.Nothing$;

import com.google.common.base.Joiner;
import com.google.common.base.Objects;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.hash.Hasher;

import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.thrift.TBase;
import org.apache.thrift.TEnum;

import com.twitter.botmaker.compiler.Serializer;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.tuples.Pair;
import com.twitter.botmaker.compiler.tuples.StructTuple;
import com.twitter.botmaker.compiler.tuples.Tuple;
import com.twitter.scrooge.ThriftStruct;
import com.twitter.util.Function;
import com.twitter.util.Function0;

public class Type {

  public static final long NON_GENERIC_TYPE_ID = 0;

  public static final int ANY_NUM_PARAMS = -1;

  private static final ImmutableSet<Class<?>> SUPPORTED_CLASSES;
  private static final ImmutableMap<Class<?>, Type> PRIMITIVE_TYPES;

  private static final ImmutableMap<String, Type> PRIMITIVE_NAME_TYPE_MAP;
  private static final ImmutableMap<String, Class<?>> COMPOSITE_NAME_CLASS_MAP;

  private static final Map<Object, Type> INTERN_MAP = Maps.newConcurrentMap();

  private static final Class<?> BOTTOM_TYPE_BASE = Nothing$.class;
  private static long nextAvailableGenericTypeId = NON_GENERIC_TYPE_ID + 1;
  private static final Pattern REGEX_MULTILINE_STR = Pattern.compile("^(.*)$", Pattern.MULTILINE);

  public static final Type OBJECT;
  public static final Type NUMBER;
  public static final Type BYTE;
  public static final Type SHORT;
  public static final Type INTEGER;
  public static final Type LONG;
  public static final Type DOUBLE;
  public static final Type BOOLEAN;
  public static final Type STRING;
  public static final Type BINARY;
  public static final Type UNIT;
  public static final Type SYMBOL;

  static {

    ImmutableSet.Builder<Class<?>> supportedClasses = ImmutableSet.builder();

    supportedClasses.add(Object.class);
    supportedClasses.add(Number.class);
    supportedClasses.add(Byte.class);
    supportedClasses.add(Short.class);
    supportedClasses.add(Integer.class);
    supportedClasses.add(Long.class);
    supportedClasses.add(Double.class);
    supportedClasses.add(Boolean.class);
    supportedClasses.add(String.class);
    supportedClasses.add(ByteBuffer.class);
    supportedClasses.add(BoxedUnit.class);
    supportedClasses.add(Collection.class);
    supportedClasses.add(List.class);
    supportedClasses.add(Set.class);
    supportedClasses.add(Map.class);
    supportedClasses.add(Pair.class);
    supportedClasses.add(Tuple.class);
    supportedClasses.add(StructTuple.class);
    supportedClasses.add(Symbol.class);

    SUPPORTED_CLASSES = supportedClasses.build();

    OBJECT = Type.of(Object.class, Serializer.OBJECT);
    NUMBER = Type.of(Number.class, Serializer.OBJECT);
    BYTE = Type.of(Byte.class, Serializer.BYTE);
    SHORT = Type.of(Short.class, Serializer.SHORT);
    INTEGER = Type.of(Integer.class, Serializer.INTEGER);
    LONG = Type.of(Long.class, Serializer.LONG);
    DOUBLE = Type.of(Double.class, Serializer.DOUBLE);
    BOOLEAN = Type.of(Boolean.class, Serializer.BOOLEAN);
    STRING = Type.of(String.class, Serializer.STRING);
    BINARY = Type.of(ByteBuffer.class, Serializer.BINARY);
    UNIT = Type.of(BoxedUnit.class, Serializer.OBJECT);
    SYMBOL = Type.of(Symbol.class, Serializer.SYMBOL);

    ImmutableMap.Builder<Class<?>, Type> primitiveTypes = ImmutableMap.builder();
    primitiveTypes.put(OBJECT.typeBase, OBJECT);
    primitiveTypes.put(NUMBER.typeBase, NUMBER);
    primitiveTypes.put(BYTE.typeBase, BYTE);
    primitiveTypes.put(SHORT.typeBase, SHORT);
    primitiveTypes.put(INTEGER.typeBase, INTEGER);
    primitiveTypes.put(LONG.typeBase, LONG);
    primitiveTypes.put(DOUBLE.typeBase, DOUBLE);
    primitiveTypes.put(BOOLEAN.typeBase, BOOLEAN);
    primitiveTypes.put(STRING.typeBase, STRING);
    primitiveTypes.put(BINARY.typeBase, BINARY);
    primitiveTypes.put(UNIT.typeBase, UNIT);

    PRIMITIVE_TYPES = primitiveTypes.build();

    ImmutableMap.Builder<String, Type> primitiveNameTypes = ImmutableMap.builder();
    primitiveNameTypes.put(OBJECT.typeBase.getSimpleName(), OBJECT);
    primitiveNameTypes.put(NUMBER.typeBase.getSimpleName(), NUMBER);
    primitiveNameTypes.put(BYTE.typeBase.getSimpleName(), BYTE);
    primitiveNameTypes.put(SHORT.typeBase.getSimpleName(), SHORT);
    primitiveNameTypes.put(INTEGER.typeBase.getSimpleName(), INTEGER);
    primitiveNameTypes.put(LONG.typeBase.getSimpleName(), LONG);
    primitiveNameTypes.put(DOUBLE.typeBase.getSimpleName(), DOUBLE);
    primitiveNameTypes.put(BOOLEAN.typeBase.getSimpleName(), BOOLEAN);
    primitiveNameTypes.put(STRING.typeBase.getSimpleName(), STRING);
    primitiveNameTypes.put(SYMBOL.typeBase.getSimpleName(), SYMBOL);
    primitiveNameTypes.put(BINARY.typeBase.getSimpleName(), BINARY);
    primitiveNameTypes.put(UNIT.typeBase.getSimpleName(), UNIT);
    primitiveNameTypes.put("Binary", BINARY);
    primitiveNameTypes.put("Unit", UNIT);

    PRIMITIVE_NAME_TYPE_MAP = primitiveNameTypes.build();

    ImmutableMap.Builder<String, Class<?>> compositeNameClasses = ImmutableMap.builder();
    compositeNameClasses.put(Collection.class.getSimpleName(), Collection.class);
    compositeNameClasses.put(List.class.getSimpleName(), List.class);
    compositeNameClasses.put(Set.class.getSimpleName(), Set.class);
    compositeNameClasses.put(Map.class.getSimpleName(), Map.class);
    compositeNameClasses.put(Pair.class.getSimpleName(), Pair.class);
    compositeNameClasses.put(Tuple.class.getSimpleName(), Tuple.class);
    compositeNameClasses.put(StructTuple.class.getSimpleName(), StructTuple.class);

    COMPOSITE_NAME_CLASS_MAP = compositeNameClasses.build();
  }

  public final Class<?> typeBase;
  public final String typeName;
  public final ImmutableList<Type> typeParams;
  public final long genericTypeId;
  public final Serializer<Object> serializer;

  private final int cachedHashCode;
  private final String cachedStringRepresentation;
  private final boolean isSupported;

  protected Type(Class<?> typeBase, String typeName, Serializer serializer) {

    this(
        typeBase,
        typeName,
        ImmutableList.of(),
        NON_GENERIC_TYPE_ID,
        serializer
    );
  }

  protected Type(
      Class<?> typeBase,
      String typeName,
      ImmutableList<Type> typeParams,
      Serializer serializer) {

    this(
        typeBase,
        typeName,
        typeParams,
        NON_GENERIC_TYPE_ID,
        serializer
    );
  }

  protected Type(
      Class<?> typeBase,
      String typeName,
      ImmutableList<Type> typeParams,
      long genericTypeId,
      Serializer serializer) {

    if (BOTTOM_TYPE_BASE.equals(typeBase)) {
      Preconditions.checkArgument(genericTypeId != NON_GENERIC_TYPE_ID);
      Preconditions.checkArgument(typeParams.isEmpty());
    } else {
      Preconditions.checkArgument(genericTypeId == NON_GENERIC_TYPE_ID);
      Integer expectedParamsCount = getClassNumParams(typeBase);
      Preconditions.checkArgument(
          expectedParamsCount == null
              || expectedParamsCount == typeParams.size()
              || expectedParamsCount == ANY_NUM_PARAMS);
    }

    this.typeBase = typeBase;
    this.typeName = typeName;
    this.typeParams = typeParams;
    this.genericTypeId = genericTypeId;
    this.serializer = serializer;

    this.cachedHashCode = computeHashCode();
    this.cachedStringRepresentation = computeStringRepresentation();

    boolean supported = true;
    if (!isClassSupported(typeBase)) {
      supported = false;
    } else {
      for (Type typeParam : typeParams) {
        if (!typeParam.isSupported) {
          supported = false;
          break;
        }
      }
    }
    this.isSupported = supported;
  }

  private int computeHashCode() {
    return new HashCodeBuilder()
        .append(typeBase)
        .append(typeParams)
        .append(genericTypeId)
        .toHashCode();
  }

  private String computeStringRepresentation() {
    if (BOTTOM_TYPE_BASE.equals(typeBase)) {
      return "t" + this.genericTypeId;
    } else if (this.typeParams.isEmpty()) {
      return this.typeName;
    } else {
      StringBuilder builder = new StringBuilder();
      builder.append(this.typeName);
      builder.append('<');
      Joiner.on(", ").appendTo(builder, this.typeParams);
      builder.append('>');
      return builder.toString();
    }
  }

  public static synchronized <T extends Type> T createType(
      Object key,
      Function0<T> createFunc
  ) {
    Type cachedType = INTERN_MAP.get(key);
    if (cachedType == null) {
      cachedType = createFunc.apply();
      INTERN_MAP.put(key, cachedType);
    }

    return (T) cachedType;
  }

  public static Type listOf(Type typeParam) {
    return Type.of(List.class, typeParam, Serializer.listOf(typeParam));
  }

  public static Type setOf(Type typeParam) {
    return Type.of(Set.class, typeParam, Serializer.setOf(typeParam));
  }

  public static Type collectionOf(Type typeParam) {
    return Type.of(Collection.class, typeParam, Serializer.collectionOf(typeParam));
  }

  public static TupleType pairOf(Type t1, Type t2) {
    return TupleType.ofTuple(
        Pair.class,
        ImmutableList.of(t1, t2),
        Serializer.pairOf(t1, t2));
  }

  public static TupleType tupleOf(Type... typeParams) {
    if (typeParams.length == 2) {
      return pairOf(typeParams[0], typeParams[1]);
    } else {
      return TupleType.ofTuple(
          Tuple.class,
          ImmutableList.copyOf(typeParams),
          Serializer.tupleOf(typeParams));
    }
  }

  public static StructTupleType structTupleOf(
      ImmutableList<String> fieldNames, ImmutableList<Type> fieldTypes) {
    return StructTupleType.ofStructTuple(fieldNames, fieldTypes);
  }

  public static Type mapOf(Type k, Type v) {
    return Type.of(Map.class, k, v, Serializer.mapOf(k, v));
  }

  public static ThriftType thriftOf(
      Class<? extends TBase> typeBase) {
    return ThriftType.ofThrift(typeBase);
  }

  public static ThriftEnumType thriftEnumOf(Class<? extends TEnum> typeBase) {
    return ThriftEnumType.ofThriftEnum(typeBase);
  }

  public static ThriftStructType thriftStructOf(
      Class<? extends ThriftStruct> typeBase) {
    return ThriftStructType.ofThriftStruct(typeBase);
  }

  static Type of(Class<?> typeBase, ImmutableList<Type> typeParams)
      throws SemanticCheckFailure {
    if (List.class.isAssignableFrom(typeBase) && typeParams.size() == 1) {
      return listOf(typeParams.get(0));
    } else if (Set.class.isAssignableFrom(typeBase) && typeParams.size() == 1) {
      return setOf(typeParams.get(0));
    } else if (Collection.class.isAssignableFrom(typeBase) && typeParams.size() == 1) {
      return collectionOf(typeParams.get(0));
    } else if (Map.class.isAssignableFrom(typeBase) && typeParams.size() == 2) {
      return mapOf(typeParams.get(0), typeParams.get(1));
    } else if (Pair.class.isAssignableFrom(typeBase) && typeParams.size() == 2) {
      return pairOf(typeParams.get(0), typeParams.get(1));
    } else if (StructTuple.class.isAssignableFrom(typeBase) && typeParams.size() > 0) {
      throw new SemanticCheckFailure(
          "Cannot create StructTuple Type without a fieldName list.");
    } else if (Tuple.class.isAssignableFrom(typeBase) && typeParams.size() > 0) {
      return tupleOf(typeParams.toArray(new Type[]{}));
    } else if (TBase.class.isAssignableFrom(typeBase) && typeParams.size() == 0) {
      return thriftOf((Class<? extends TBase>) typeBase);
    } else if (TEnum.class.isAssignableFrom(typeBase) && typeParams.size() == 0) {
      return thriftEnumOf((Class<? extends TEnum>) typeBase);
    } else if (ThriftStruct.class.isAssignableFrom(typeBase) && typeParams.size() == 0) {
      return thriftStructOf((Class<? extends ThriftStruct>) typeBase);
    } else if (ByteBuffer.class.isAssignableFrom(typeBase) && typeParams.size() == 0) {
      return Type.BINARY;
    } else if (PRIMITIVE_TYPES.containsKey(typeBase) && typeParams.size() == 0) {
      return PRIMITIVE_TYPES.get(typeBase);
    } else {
      return Type.of(typeBase, typeParams, Serializer.OBJECT);
    }
  }

  public static synchronized Type newGenericType() {
    long genericTypeId = nextAvailableGenericTypeId++;

    return new Type(
        BOTTOM_TYPE_BASE,
        "t" + genericTypeId,
        ImmutableList.of(),
        genericTypeId,
        Serializer.OBJECT);
  }

  public static boolean doubleLike(Number number) {
    return number instanceof Float
        || number instanceof Double;
  }

  public static boolean longLike(Number number) {
    return number instanceof Long
        || number instanceof Integer
        || number instanceof Short
        || number instanceof Byte;
  }

  public static boolean isClassSupported(Class<?> clazz) {
    return SUPPORTED_CLASSES.contains(clazz)
        || TBase.class.isAssignableFrom(clazz)
        || ThriftStruct.class.isAssignableFrom(clazz)
        || TEnum.class.isAssignableFrom(clazz);
  }

  public static int getClassNumParams(Class<?> classz) {
    if (Pair.class.isAssignableFrom(classz)) {
      return 2;
    } else if (StructTuple.class.isAssignableFrom(classz)) {
      return ANY_NUM_PARAMS;
    } else if (Tuple.class.isAssignableFrom(classz)) {
      return ANY_NUM_PARAMS;
    } else if (Map.class.isAssignableFrom(classz)) {
      return 2;
    } else if (Collection.class.isAssignableFrom(classz)) {
      return 1;
    } else {
      return 0;
    }
  }

  private static Type of(Class<?> typeBase, Serializer serializer) {
    return Type.of(typeBase, ImmutableList.<Type>of(), serializer);
  }

  private static Type of(Class<?> typeBase, Type typeParam1, Serializer serializer) {
    return Type.of(typeBase, ImmutableList.of(typeParam1), serializer);
  }

  private static Type of(
      Class<?> typeBase,
      Type typeParam1,
      Type typeParam2,
      Serializer serializer) {
    return Type.of(typeBase, ImmutableList.of(typeParam1, typeParam2), serializer);
  }

  private static Type of(
      Class<?> typeBase,
      ImmutableList<Type> typeParams,
      Serializer serializer) {
    Preconditions.checkArgument(isClassSupported(typeBase));
    return createType(
        Pair.of(typeBase, typeParams),
        Function.func0(() ->
            new Type(
                typeBase,
                typeBase.getSimpleName(),
                typeParams,
                NON_GENERIC_TYPE_ID,
                serializer)
        ));
  }

  private static Class<?> getClosestCommonSuperClass(Class<?> c1, Class<?> c2) {
    Preconditions.checkArgument(isClassSupported(c1));
    Preconditions.checkArgument(isClassSupported(c2));

    if (c1.equals(c2)) {
      return c1;
    }
    if (Collection.class.isAssignableFrom(c1)
        && Collection.class.isAssignableFrom(c2)) {
      return Collection.class;
    }
    if (Number.class.isAssignableFrom(c1)
        && Number.class.isAssignableFrom(c2)) {
      return Number.class;
    }
    if (TBase.class.isAssignableFrom(c1)
        && TBase.class.isAssignableFrom(c2)) {
      return Object.class;
    }
    if (ThriftStruct.class.isAssignableFrom(c1)
        && ThriftStruct.class.isAssignableFrom(c2)) {
      return Object.class;
    }
    if (Tuple.class.isAssignableFrom(c1)
        && Tuple.class.isAssignableFrom(c2)) {
      return Tuple.class;
    }
    return Object.class;
  }


  public static Type getClosestCommonSuperType(Type typeA, Type typeB) throws SemanticCheckFailure {
    Preconditions.checkArgument(!typeA.isGenericType());
    Preconditions.checkArgument(!typeB.isGenericType());

    if (typeA.equals(typeB)) {
      return typeA;
    }

    if (!typeA.isSupported || !typeB.isSupported) {
      return Type.OBJECT;
    }

    Class<?> commonBaseType = getClosestCommonSuperClass(typeA.typeBase, typeB.typeBase);
    ImmutableList<Type> tp1TypeParams = typeA.getTypeParams();
    ImmutableList<Type> tp2TypeParams = typeB.getTypeParams();
    int numParams = getClassNumParams(commonBaseType);
    if (numParams == ANY_NUM_PARAMS) {
      if (tp1TypeParams.size() != tp2TypeParams.size()) {
        return Type.OBJECT;
      }
      numParams = tp1TypeParams.size();
    }

    List<Type> typeParams = Lists.newArrayListWithCapacity(numParams);
    for (int i = 0; i < numParams; i++) {
      typeParams.add(getClosestCommonSuperType(
          tp1TypeParams.get(i),
          tp2TypeParams.get(i)
      ));
    }
    return Type.of(commonBaseType, ImmutableList.copyOf(typeParams));
  }

  public static boolean isDivergentTo(Class<?> c1, Class<?> c2) {
    Preconditions.checkArgument(isClassSupported(c1));
    Preconditions.checkArgument(isClassSupported(c2));

    return !c1.isAssignableFrom(c2)
        && !c2.isAssignableFrom(c1);
  }

  private static boolean isSuperOf(Class<?> c1, Class<?> c2) {
    return isClassSupported(c1)
        && isClassSupported(c2)
        && c1.isAssignableFrom(c2);
  }

  public static boolean isSuperOf(Type t1, Type t2) {
    if (!t1.isSupported
        || !t2.isSupported
        || !isSuperOf(t1.typeBase, t2.typeBase)) {
      return false;
    }
    List<Type> list1 = t1.getTypeParams();
    List<Type> list2 = t2.getTypeParams();
    for (int index = Math.min(list1.size(), list2.size()) - 1; index >= 0; --index) {
      if (!isSuperOf(list1.get(index), list2.get(index))) {
        return false;
      }
    }
    return true;
  }

  public static boolean isDivergentTo(Type typeA, Type typeB) {
    Preconditions.checkNotNull(typeA);
    Preconditions.checkNotNull(typeB);
    Preconditions.checkArgument(typeA.isSupported && typeB.isSupported);

    if (typeA == Type.OBJECT || typeB == Type.OBJECT) {
      return false;
    }
    if (isDivergentTo(typeA.typeBase, typeB.typeBase)) {
      return true;
    }
    ImmutableList<Type> typeAParams = typeA.getTypeParams();
    ImmutableList<Type> typeBParams = typeB.getTypeParams();
    int numElems = -1 + Math.max(
        typeA.getTypeParams().size(),
        typeB.getTypeParams().size());
    for (int i = numElems; i >= 0; i--) {
      if (isDivergentTo(typeAParams.get(i), typeBParams.get(i))) {
        return true;
      }
    }
    return false;
  }

  public final void computerFingerprint(Hasher hasher) {

    hasher.putUnencodedChars(typeBase.getName());
    hasher.putLong(genericTypeId);

    hasher.putInt(typeParams.size());
    for (Type type : typeParams) {
      type.computerFingerprint(hasher);
    }
  }

  public Type replaceTypeArguments(Map<Long, Type> mappedTypes) throws SemanticCheckFailure {
    if (this.isGenericType()) {
      if (mappedTypes.containsKey(genericTypeId)) {
        return mappedTypes.get(this.genericTypeId);
      } else {
        return Type.OBJECT;
      }
    } else {
      ImmutableList<Type> params = this.getTypeParams();
      List<Type> inferredTypeParams = Lists.newArrayListWithCapacity(params.size());
      for (Type typeParam : params) {
        Type replacement = typeParam.replaceTypeArguments(mappedTypes);
        inferredTypeParams.add(replacement);
      }
      return Type.of(this.typeBase, ImmutableList.copyOf(inferredTypeParams));
    }
  }

  public boolean isSupported() {
    return isSupported;
  }

  public boolean isGenericType() {
    return this.genericTypeId != NON_GENERIC_TYPE_ID;
  }

  public boolean isPrimitive() {
    return PRIMITIVE_TYPES.containsKey(this.typeBase);
  }

  public boolean stringLike() {
    return typeBase == String.class;
  }

  public boolean longLike() {
    return typeBase == Long.class
        || typeBase == Integer.class
        || typeBase == Short.class
        || typeBase == Byte.class;
  }

  public boolean doubleLike() {
    return typeBase == Double.class
        || typeBase == Float.class;
  }

  public boolean numberLike() {
    return Number.class.isAssignableFrom(typeBase) && typeParams.size() == 0;
  }

  public boolean listLike() {
    return List.class.isAssignableFrom(typeBase);
  }

  public boolean setLike() {
    return Set.class.isAssignableFrom(typeBase);
  }

  public boolean mapLike() {
    return Map.class.isAssignableFrom(typeBase);
  }

  public boolean pairLike() {
    return Pair.class.isAssignableFrom(typeBase);
  }

  public boolean structTupleLike() {
    return this instanceof StructTupleType;
  }

  public boolean thriftLike() {
    return this instanceof ThriftType;
  }

  public boolean thriftStructLike() {
    return this instanceof ThriftStructType;
  }

  public ImmutableList<Type> getTypeParams() {
    Preconditions.checkArgument(!this.isGenericType());
    return this.typeParams;
  }

  public Class<?> getBase() {
    return this.typeBase;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    return (other instanceof Type)
        && Objects.equal(this.typeBase, ((Type) other).typeBase)
        && this.genericTypeId == ((Type) other).genericTypeId
        && this.typeParams.equals(((Type) other).typeParams);
  }

  @Override
  public int hashCode() {
    return cachedHashCode;
  }

  @Override
  public String toString() {
    return cachedStringRepresentation;
  }

  public String toMultiLineString() {
    StringBuilder result = new StringBuilder();
    result.append(String.format("%s\n", typeBase.getSimpleName()));
    for (Type param : typeParams) {
      final String str = param.toMultiLineString();
      result.append(REGEX_MULTILINE_STR.matcher(str).replaceAll("  $1"));
    }
    return result.toString();
  }

  public static Optional<Pair<Class<?>, Integer>> getCompositeClassInfoFromName(String name) {
    if (!COMPOSITE_NAME_CLASS_MAP.containsKey(name)) {
      return Optional.absent();
    }
    Class<?> clazz = COMPOSITE_NAME_CLASS_MAP.get(name);
    Integer numParams = getClassNumParams(clazz);
    return Optional.of(Pair.<Class<?>, Integer>of(clazz, numParams));
  }

  public static Optional<Type> getPrimitiveTypeFromName(String name) {
    if (!PRIMITIVE_NAME_TYPE_MAP.containsKey(name)) {
      return Optional.absent();
    }
    return Optional.of(PRIMITIVE_NAME_TYPE_MAP.get(name));
  }
}
