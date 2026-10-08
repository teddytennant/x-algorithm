package com.twitter.botmaker.function.thrift;

import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.tuples.Pair;
import com.twitter.botmaker.compiler.types.ThriftStructType;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.Constant;
import com.twitter.botmaker.function.FunctionNode;
import com.twitter.botmaker.function.collection.ListOperator;
import com.twitter.botmaker.function.collection.MapOperator;
import com.twitter.botmaker.function.collection.SetOperator;
import com.twitter.botmaker.function.conversion.ToPair;
import com.twitter.scrooge.ThriftStruct;

public abstract class ThriftStructOperator<E, T extends ThriftStruct> extends FunctionNode<E> {

  private final ThriftStructType thriftType;

  public ThriftStructOperator(
      Class<T> thriftClass,
      String exprText, ImmutableList<ASTNode> children)
      throws SemanticCheckFailure {
    super(exprText, children);
    this.thriftType = Type.thriftStructOf(thriftClass);
    checkThriftFieldTypes(this.thriftType, this.getChildren());
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Global;
  }

  @Override
  @SuppressWarnings("unchecked")
  public Object apply(Context<E> context, List<Object> args) throws Exception {

    Map<Object, Object> hashMap = Maps.newHashMap();
    for (Object arg : args) {
      Pair<?, ?> pair = (Pair<?, ?>) arg;
      Object key = pair.getFirst();
      Object value = pair.getSecond();
      if (key != null && value != null) {
        hashMap.put(key, value);
      }
    }

    T thriftObject = (T) thriftType.newInstance(hashMap);
    return apply(context, thriftObject);
  }

  public static void checkThriftFieldTypes(
      ThriftStructType type, List<ASTNode> nodes) throws SemanticCheckFailure {
    for (ASTNode node : nodes) {
      if (!(node instanceof ToPair)) {
        continue;
      }

      ToPair pair = (ToPair) node;
      ASTNode first = pair.getChildren().get(0);
      ASTNode second = pair.getChildren().get(1);

      if (!(first instanceof Constant)) {
        continue;
      }

      Object value = ((Constant) first).getValue();
      if (!(value instanceof String)) {
        throw new SemanticCheckFailure(
            String.format("field name %s has to be a String.", value)
        );
      }

      String fieldName = (String) value;
      Type fieldType = type.getFieldType(fieldName);
      if (isDivergentTo(fieldType, second)) {
        throw new SemanticCheckFailure(
            String.format("field %s expects %s type but %s received.",
                fieldName, fieldType, second.getReturnType()));
      }
    }
  }

  private static boolean isDivergentTo(
      Type fieldType, ASTNode<?> node) throws SemanticCheckFailure {

    if (fieldType.typeBase == Set.class && node instanceof SetOperator) {
      Type elementType = fieldType.getTypeParams().get(0);
      for (ASTNode child : node.getChildren()) {
        if (isDivergentTo(elementType, child)) {
          return true;
        }
      }
      return false;
    }

    if (fieldType.typeBase == List.class && node instanceof ListOperator) {
      Type elementType = fieldType.getTypeParams().get(0);
      for (ASTNode child : node.getChildren()) {
        if (isDivergentTo(elementType, child)) {
          return true;
        }
      }
      return false;
    }

    if (fieldType.typeBase == Map.class && node instanceof MapOperator) {
      Type keyType = fieldType.getTypeParams().get(0);
      Type valueType = fieldType.getTypeParams().get(1);
      for (ASTNode child : node.getChildren()) {
        ToPair pair = (ToPair) child;
        if (isDivergentTo(keyType, pair.getChildren().get(0))) {
          return true;
        }
        if (isDivergentTo(valueType, pair.getChildren().get(1))) {
          return true;
        }
      }
      return false;
    }

    if (fieldType instanceof ThriftStructType && node instanceof MapOperator) {
      checkThriftFieldTypes((ThriftStructType) fieldType, node.getChildren());
      return false;
    }

    return Type.isDivergentTo(fieldType, node.getReturnType());
  }

  protected abstract Object apply(Context<E> context, T thriftObject) throws Exception;
}
