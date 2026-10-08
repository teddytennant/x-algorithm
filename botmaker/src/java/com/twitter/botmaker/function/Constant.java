package com.twitter.botmaker.function;

import java.nio.ByteBuffer;
import java.util.List;

import scala.runtime.BoxedUnit;

import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.Extractor;
import com.twitter.botmaker.Extractor.Functor;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Symbol;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.runtime.Runtime;
import com.twitter.util.Future;

public abstract class Constant extends ASTNode<Runtime> {
  private final Object value;
  private final Future<Object> futureValue;

  private static final Signature NULL_SIGNATURE;
  private static final Signature LONG_SIGNATURE;
  private static final Signature DOUBLE_SIGNATURE;
  private static final Signature STRING_SIGNATURE;
  private static final Signature BOOLEAN_SIGNATURE;
  private static final Signature LIST_LONG_SIGNATURE;
  private static final Signature LIST_DOUBLE_SIGNATURE;
  private static final Signature BINARY_SIGNATURE;
  private static final Signature UNIT_SIGNATURE;
  private static final Signature SYMBOL_SIGNATURE;

  public static final Constant NULL;
  public static final Constant UNIT;

  static {
    ImmutableList<Type> emptyList = ImmutableList.of();
    NULL_SIGNATURE = new Signature(emptyList, Type.OBJECT);
    LONG_SIGNATURE = new Signature(emptyList, Type.LONG);
    DOUBLE_SIGNATURE = new Signature(emptyList, Type.DOUBLE);
    STRING_SIGNATURE = new Signature(emptyList, Type.STRING);
    BOOLEAN_SIGNATURE = new Signature(emptyList, Type.BOOLEAN);
    LIST_LONG_SIGNATURE = new Signature(emptyList, Type.listOf(Type.LONG));
    LIST_DOUBLE_SIGNATURE = new Signature(emptyList, Type.listOf(Type.DOUBLE));
    BINARY_SIGNATURE = new Signature(emptyList, Type.BINARY);
    UNIT_SIGNATURE = new Signature(emptyList, Type.UNIT);
    SYMBOL_SIGNATURE = new Signature(emptyList, Type.SYMBOL);
    try {
      NULL = new Constant("NULL", null) {
        @Override
        public Signature getSignature() {
          return NULL_SIGNATURE;
        }
      };
    } catch (SemanticCheckFailure e) {
      throw new RuntimeException("Failed to make BotMaker NULL constant", e);
    }
    try {
      UNIT = new Constant("UNIT", BoxedUnit.UNIT) {
        @Override
        public Signature getSignature() {
          return UNIT_SIGNATURE;
        }
      };
    } catch (SemanticCheckFailure e) {
      throw new RuntimeException("Failed to make BotMaker UNIT constant", e);
    }
  }

  private Constant(String exprText, Object value) throws SemanticCheckFailure {
    super(exprText, ImmutableList.of());
    this.value = value;
    this.futureValue = Future.value(value);
  }

  public static Constant of(String exprText, Long longValue) throws SemanticCheckFailure {
    return new Constant(exprText, longValue) {
      @Override
      public Signature getSignature() {
        return LONG_SIGNATURE;
      }
    };
  }

  public static Constant of(String exprText, Double doubleValue) throws SemanticCheckFailure {
    return new Constant(exprText, doubleValue) {
      @Override
      public Signature getSignature() {
        return DOUBLE_SIGNATURE;
      }
    };
  }

  public static Constant of(String exprText, String stringValue) throws SemanticCheckFailure {
    String stringValueInterned = intern(stringValue);
    return new Constant(exprText, stringValueInterned) {
      @Override
      public Signature getSignature() {
        return STRING_SIGNATURE;
      }
    };
  }

  public static Constant of(String exprText, Symbol symbol) throws SemanticCheckFailure {
    return new Constant(exprText, symbol) {
      @Override
      public Signature getSignature() {
        return SYMBOL_SIGNATURE;
      }
    };
  }

  public static Constant of(String exprText, Boolean booleanValue) throws SemanticCheckFailure {
    return new Constant(exprText, booleanValue) {
      @Override
      public Signature getSignature() {
        return BOOLEAN_SIGNATURE;
      }
    };
  }

  public static Constant ofDoubleList(
      String exprText,
      List<Double> doubleList) throws SemanticCheckFailure {
    return new Constant(exprText, doubleList) {
      @Override
      public Signature getSignature() {
        return LIST_DOUBLE_SIGNATURE;
      }
    };
  }

  public static Constant ofLongList(
      String exprText,
      List<Long> longList) throws SemanticCheckFailure {
    return new Constant(exprText, longList) {
      @Override
      public Signature getSignature() {
        return LIST_LONG_SIGNATURE;
      }
    };
  }

  public static Constant of(
      String exprText,
      ByteBuffer binary) throws SemanticCheckFailure {
    return new Constant(exprText, binary) {
      @Override
      public Signature getSignature() {
        return BINARY_SIGNATURE;
      }
    };
  }

  public Object getValue() {
    return value;
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Global;
  }

  @Override
  public Extractor<Runtime> toExtractor() {
    return new Functor<Runtime>(this, ImmutableList.of()) {
      @Override
      public Future run(Context<Runtime> context) {
        return futureValue;
      }

      @Override
      public Object apply(Context<Runtime> context) {
        return value;
      }
    };
  }

}
